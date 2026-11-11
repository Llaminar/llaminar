/**
 * @file PipelineNativeOwnership.h
 * @brief Authenticate every native local pipeline member against its original rank publication.
 *
 * Discovery visibility does not grant execution ownership. Canonically selected
 * execution contexts retain their discovery owner/membership; this check uses
 * that exact supplied context's published physical identities and the original
 * prepared worker's UUID rather than reconstructing or filtering an inventory.
 */
#pragma once
#include "interfaces/IMPIContext.h"
#include "planning/RankHardwareOwnership.h"
#include <span>
#include <algorithm>
#include <string>
#include <set>
namespace llaminar2 {
/** @brief Reject remote, unknown, repeated or substituted original GPU members before creating a native context.
 * @param original Actual parent context retained from runtime admission, including canonical rank selection.
 * @param physical_rank Final PMA rank in that same execution namespace.
 * @param devices Every original stage member, not just stage leaders.
 * @param prepared_uuids UUIDs from those exact original prepared worker birth keys.
 */
inline void requireOriginalNativePipelineOwnership(const IMPIContext &original, int physical_rank,
    std::span<const DeviceId> devices, std::span<const std::string> prepared_uuids)
{
    const auto inventory = original.clusterInventory();
    if (!inventory || original.rank() < 0 || original.rank() >= original.world_size() ||
        physical_rank != original.rank() || inventory->world_size != original.world_size() ||
        inventory->ranks.size() != static_cast<std::size_t>(original.world_size()) ||
        devices.size() < 2 || prepared_uuids.size() != devices.size())
        throw std::invalid_argument("Native pipeline requires its exact original MPI/PMA rank authority");
    const auto &rank = inventory->getRank(original.rank());
    if (rank.rank != original.rank())
        throw std::invalid_argument("Native pipeline inventory changed its original rank namespace");
    std::set<std::pair<DeviceType, int>> physical;
    std::set<std::string> uuids;
    for (std::size_t member = 0; member < devices.size(); ++member) {
        const auto device = devices[member];
        requireRankLocalGPUOwnership(rank, device, "native pipeline original member");
        const auto observed = std::find_if(rank.gpus.begin(), rank.gpus.end(), [&](const auto &gpu) {
            return gpu.type == device.type && gpu.local_device_id == device.ordinal;
        });
        if (observed == rank.gpus.end() || observed->uuid.empty() || prepared_uuids[member] != observed->uuid ||
            !physical.emplace(device.type, device.ordinal).second || !uuids.insert(observed->uuid).second)
            throw std::invalid_argument("Native pipeline member changed its original physical UUID/prepared birth");
    }
}
} // namespace llaminar2
