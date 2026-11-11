/**
 * @file OriginalMappedNativeCohorts.h
 * @brief Immutable real-rank cohort binding for explicit mapped native components.
 *
 * A binding borrows the original discovery publication, preserves physical
 * UUID/NUMA ownership and chooses distinct original ranks of one host. It owns
 * no worker, allocation, communicator or execution state. Native callers must
 * prepare only localDevices(rank) through their original source/PMA lifecycle.
 */
#pragma once
#include "fixtures/backends/OriginalRankLocalNativeCohort.h"
#include <limits>
#include <memory>
#include <span>
#include <utility>
#include <vector>
#include <set>

namespace llaminar2::test {
/** Exact positive original native domain request; counts never imply rank or ordinal. */
struct OriginalMappedNativeDomainRequest {
    DeviceType backend;
    std::size_t degree;
};
/** Original source/peer binding, immutable after observed cohort selection. */
class OriginalMappedNativeCohorts final {
public:
    /**
     * @brief Bind both complete domains from an existing immutable publication.
     * @param inventory Same complete original MPI hardware publication used by admission.
     * @param source Required original source family and positive width.
     * @param peer Required original peer family and positive width.
     * @return Distinct NUMA-local original owners on one actual host.
     * @throws std::invalid_argument For invalid geometry/publication or physical UUID aliases.
     * @throws std::runtime_error When no original rank pair owns this declared geometry.
     */
    static OriginalMappedNativeCohorts observe(std::shared_ptr<const ClusterInventory> inventory,
        OriginalMappedNativeDomainRequest source,OriginalMappedNativeDomainRequest peer) {
        const auto valid=[](const auto& request){return (request.backend==DeviceType::CUDA || request.backend==DeviceType::ROCm) &&
            request.degree>0 && request.degree<=std::size_t(std::numeric_limits<int>::max());};
        if(!inventory || !valid(source) || !valid(peer) || inventory->world_size<=0 ||
            inventory->ranks.size()!=std::size_t(inventory->world_size))
            throw std::invalid_argument("Mapped native cohorts require original complete inventory and positive int-representable domains");
        for(int rank=0;rank<inventory->world_size;++rank)if(inventory->ranks[rank].rank!=rank)
            throw std::invalid_argument("Mapped native cohorts cannot relabel original discovery ranks");
        const auto select=[](const RankInventory& rank,const auto& request){
            std::vector<DeviceId> devices;
            for(const auto& gpu:rank.gpus)if(gpu.type==request.backend && rankOwnsGPUDevice(rank,gpu))
                devices.emplace_back(gpu.type,gpu.local_device_id);
            if(devices.size()<request.degree)return std::vector<DeviceId>{};
            devices.resize(request.degree);return devices;
        };
        for(const auto& source_rank:inventory->ranks){const auto source_devices=select(source_rank,source);if(source_devices.empty())continue;
            for(const auto& peer_rank:inventory->ranks){
                if(source_rank.rank==peer_rank.rank || source_rank.cpu.numa_node==peer_rank.cpu.numa_node ||
                    inventory->connectionBetweenRanks(source_rank.rank,peer_rank.rank).locality()!=RankConnectionLocality::SameNode)continue;
                const auto peer_devices=select(peer_rank,peer);if(peer_devices.empty())continue;
                (void)requireOriginalRankLocalNativeCohort(*inventory,source_rank.rank,inventory->world_size,source_rank.cpu.numa_node,source_devices);
                (void)requireOriginalRankLocalNativeCohort(*inventory,peer_rank.rank,inventory->world_size,peer_rank.cpu.numa_node,peer_devices);
                std::set<std::string> uuids;
                for(const auto role:{std::pair{&source_rank,&source_devices},std::pair{&peer_rank,&peer_devices}})
                    for(const auto device:*role.second){const auto found=std::find_if(role.first->gpus.begin(),role.first->gpus.end(),[&](const auto& gpu){return gpu.type==device.type && gpu.local_device_id==device.ordinal;});
                        if(!uuids.insert(found->uuid).second)throw std::invalid_argument("Mapped native domains cannot alias a physical GPU UUID");}
                return OriginalMappedNativeCohorts(std::move(inventory),source_rank.rank,peer_rank.rank,source_devices,peer_devices);
            }
        }
        throw std::runtime_error("Declared mapped native domains are absent from two distinct original NUMA-local ranks");
    }
    /** @return The same authentic physical pair with source/peer roles exchanged. */
    OriginalMappedNativeCohorts reversed()const {return OriginalMappedNativeCohorts(inventory_,peer_rank_,source_rank_,peer_devices_,source_devices_);}
    /** @return Actual original source rank, never a socket or ordinal alias. */
    int sourceRank()const noexcept{return source_rank_;}
    /** @return Actual original peer rank. */
    int peerRank()const noexcept{return peer_rank_;}
    /** @return Original ordered selected source endpoints. */
    const std::vector<DeviceId>& sourceDevices()const noexcept{return source_devices_;}
    /** @return Original ordered selected peer endpoints. */
    const std::vector<DeviceId>& peerDevices()const noexcept{return peer_devices_;}
    /** @return Only endpoints truly owned by this original rank; idle ranks remain empty. */
    std::span<const DeviceId> localDevices(int rank)const noexcept {
        if(rank==source_rank_)return source_devices_;if(rank==peer_rank_)return peer_devices_;return {};
    }
    /** @return Same original immutable inventory, without narrowing or reconstruction. */
    const std::shared_ptr<const ClusterInventory>& inventory()const noexcept{return inventory_;}
private:
    /** Preserve a completed authenticated selection; no public unchecked constructor exists. */
    OriginalMappedNativeCohorts(std::shared_ptr<const ClusterInventory> inventory,int source,int peer,
        std::vector<DeviceId> source_devices,std::vector<DeviceId> peer_devices):inventory_(std::move(inventory)),
        source_rank_(source),peer_rank_(peer),source_devices_(std::move(source_devices)),peer_devices_(std::move(peer_devices)){}
    std::shared_ptr<const ClusterInventory> inventory_;
    int source_rank_,peer_rank_;
    std::vector<DeviceId> source_devices_,peer_devices_;
};
}
