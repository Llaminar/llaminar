/**
 * @file ObservedISAPipelinePlacement.h
 * @brief Exact original-inventory ISA eligibility for arbitrary local pipeline domains.
 *
 * This cold declaration preserves physical UUID/NUMA/ISA ownership for any authored stage cardinalities. It neither
 * probes devices nor creates transport or memory authority. Production rank
 * selection subsequently admits the actual owner from the original discovery
 * world; ordinal adjacency never substitutes for architecture or PCI locality.
 */
#pragma once
#include "planning/ObservedPeerGrouping.h"
#include "config/OrchestrationPlanningPolicy.h"
#include <span>
#include <tuple>
#include <map>
#include <algorithm>
#include <optional>
#include <stdexcept>
#include <vector>
#include <limits>
#include <set>
#include <string>
#include <string_view>

namespace llaminar2::test {
/** @brief Immutable hardware request derived from one complete publication. */
struct ObservedISAPipelinePlacement final {
    int discovery_owner; ///< Actual rank owning the requested physical ISA cohort.
    std::vector<std::vector<DeviceId>> stages; ///< Authored positive domain sizes, with disjoint observed peer groups.

    /** @return Existing production discovery-to-execution selection. */
    ExecutionRankSelection selection() const { return ExecutionRankSelection({discovery_owner}); }

    /**
     * @brief Bind positive domain sizes to one observed original NUMA owner.
     * @param inventory Original discovery publication, never a new query.
     * @param backend Authored GPU backend, never inferred from architecture spelling.
     * @param native_isa Exact architecture base, without optional driver feature suffixes.
     * @param sizes Authored stage cardinalities, independent of device count/ordinals.
     * @return Arbitrary disjoint native domains on one original NUMA owner.
     * @throws std::invalid_argument For absent/ambiguous ISA, UUID, NUMA or cohort geometry.
     */
    static ObservedISAPipelinePlacement require(const ClusterInventory& inventory,
        DeviceType backend, std::string_view native_isa, std::span<const std::size_t> sizes) {
        if(inventory.world_size<=0 || inventory.ranks.size()!=static_cast<std::size_t>(inventory.world_size) ||
            backend == DeviceType::CPU || native_isa.empty() || sizes.empty())
            throw std::invalid_argument("ISA pipeline requires complete original inventory and positive stage geometry");
        std::size_t required=0;
        for(const auto size:sizes) {
            if(!size || size>std::numeric_limits<std::size_t>::max()-required)
                throw std::invalid_argument("ISA pipeline stage geometry is empty or overflows");
            required+=size;
        }
        std::optional<ObservedISAPipelinePlacement> result;
        for(const auto& rank:inventory.ranks) {
            std::vector<DeviceId> eligible;
            std::set<std::string> uuids;
            for(const auto& gpu:rank.gpus) {
                const std::string_view isa(gpu.native_isa);
                if(gpu.type!=backend || !rankOwnsGPUDevice(rank,gpu) ||
                    isa.substr(0,isa.find(':'))!=native_isa) continue;
                if(gpu.uuid.empty() || !uuids.insert(gpu.uuid).second)
                    throw std::invalid_argument("ISA pipeline requires distinct observed GPU UUIDs");
                eligible.emplace_back(gpu.type,gpu.local_device_id);
            }
            if(eligible.size()<required) continue;
            if(result)
                throw std::invalid_argument("ISA pipeline request is ambiguous across original NUMA owners");
            result=ObservedISAPipelinePlacement{rank.rank,ObservedPeerGrouping(rank).select(eligible,sizes)};
        }
        if(!result) throw std::invalid_argument("ISA pipeline has no complete original NUMA-owned architecture cohort");
        return *result;
    }
};
/** @brief One authored native stage; physical participants remain original observations. */
struct ObservedNativePipelineStage final {
    DeviceType backend; ///< Native homogeneous transport within this stage.
    std::size_t width; ///< Positive participant count, without a machine-specific cap.
    std::string native_isa{}; ///< Optional exact architecture base; observed feature suffixes are excluded.
};

/** @brief Complete cross-vendor placement across distinct original owners on one physical host. */
struct ObservedCrossVendorPipelinePlacement final {
    std::vector<int> owners; ///< Original discovery owner per stage; repeated backends retain their real owner.
    std::vector<std::vector<DeviceId>> stages; ///< Disjoint original UUID-authenticated participants in authored order.

    /** @return Existing production rank selection in first-stage ownership order, without inventing ranks. */
    ExecutionRankSelection selection() const {
        std::vector<int> unique;
        for(const auto owner:owners)
            if(std::find(unique.begin(),unique.end(),owner)==unique.end())unique.push_back(owner);
        return ExecutionRankSelection(std::move(unique));
    }

    /**
     * @brief Solve the complete ordered request before choosing owners or GPUs.
     * @param inventory Complete original MPI publication, never a new native query.
     * @param request Positive CUDA/ROCm stages with optional per-stage architecture constraints.
     * @return Best complete distinct-owner assignment with peer-aware disjoint groups.
     * @throws std::invalid_argument For missing placement or malformed/contradictory physical identity.
     *
     * Repeated stages consume their combined demand, including overlapping ISA
     * eligibility. A small opposite-vendor cohort on the first socket cannot
     * steal a stage from the complete assignment. Native transport remains
     * homogeneous and rank-local; pipeline boundaries keep original MPI owners.
     */
    static ObservedCrossVendorPipelinePlacement require(const ClusterInventory& inventory,
        std::span<const ObservedNativePipelineStage> request) {
        if(inventory.world_size<=0 || inventory.ranks.size()!=static_cast<std::size_t>(inventory.world_size) ||
            request.size()<2)
            throw std::invalid_argument("Cross-vendor pipeline requires its complete original world and stages");
        std::map<DeviceType,std::size_t> totals;
        for(const auto& stage:request) {
            if((stage.backend!=DeviceType::CUDA && stage.backend!=DeviceType::ROCm) || !stage.width ||
                stage.width>std::numeric_limits<std::size_t>::max()-totals[stage.backend] ||
                stage.native_isa.find(':')!=std::string::npos)
                throw std::invalid_argument("Cross-vendor stages require positive native geometry and optional exact ISA bases");
            totals[stage.backend]+=stage.width;
        }
        if(totals.size()!=2)
            throw std::invalid_argument("Cross-vendor pipeline must explicitly declare both native backends");
        std::map<DeviceType,std::map<int,std::vector<DeviceId>>> eligible;
        std::set<int> ranks;
        using PhysicalFacts=std::tuple<DeviceType,int,std::string,std::vector<std::string>>;
        std::map<std::pair<std::string,std::string>,PhysicalFacts> physical_facts;
        for(const auto& rank:inventory.ranks) {
            if(rank.rank<0 || rank.rank>=inventory.world_size || !ranks.insert(rank.rank).second)
                throw std::invalid_argument("Cross-vendor pipeline lost original rank identity");
            std::set<std::string> uuids;
            std::set<DeviceId> local_ids;
            for(const auto& gpu:rank.gpus) {
                if(!totals.contains(gpu.type) || !rankOwnsGPUDevice(rank,gpu))continue;
                if(gpu.local_device_id<0 || gpu.uuid.empty() || !uuids.insert(gpu.uuid).second ||
                    !local_ids.insert(DeviceId(gpu.type,gpu.local_device_id)).second)
                    throw std::invalid_argument("Cross-vendor pipeline requires distinct original GPU identities");
                const PhysicalFacts facts{gpu.type,gpu.numa_node,gpu.native_isa,gpu.pcie_upstream_bdfs};
                const auto [found,inserted]=physical_facts.emplace(std::pair{rank.hostname,gpu.uuid},facts);
                if(!inserted && found->second!=facts)
                    throw std::invalid_argument("Original cross-vendor UUID has contradictory backend/NUMA/ISA/PCI identity");
                eligible[gpu.type][rank.rank].emplace_back(gpu.type,gpu.local_device_id);
            }
        }
        std::optional<ObservedCrossVendorPipelinePlacement> result;
        ObservedPeerGroupQuality best_quality{};
        using OriginalIdentity=std::tuple<std::string,int,int,int,std::string,DeviceType,int>;
        std::vector<OriginalIdentity> best_identity;
        for(const auto& cuda:inventory.ranks) for(const auto& rocm:inventory.ranks) {
            if(cuda.rank==rocm.rank || cuda.node_id<0 || cuda.node_id!=rocm.node_id ||
                cuda.hostname.empty() || cuda.hostname!=rocm.hostname ||
                eligible[DeviceType::CUDA][cuda.rank].size()<totals[DeviceType::CUDA] ||
                eligible[DeviceType::ROCm][rocm.rank].size()<totals[DeviceType::ROCm])continue;
            ObservedCrossVendorPipelinePlacement candidate;
            candidate.stages.resize(request.size());
            for(const auto& stage:request)candidate.owners.push_back(stage.backend==DeviceType::CUDA?cuda.rank:rocm.rank);
            bool fits=true;
            ObservedPeerGroupQuality quality{};
            // Backend pools have independent physical identity, so each
            // original owner's optimum maximizes the additive whole-request
            // quality. Canonical grouping alone solves overlapping stage filters.
            for(const auto backend:{DeviceType::CUDA,DeviceType::ROCm}) {
                const auto& owner=backend==DeviceType::CUDA?cuda:rocm;
                const ObservedPeerGrouping grouping(owner);
                std::vector<std::size_t> stage_indices;
                std::vector<std::vector<DeviceId>> candidates;
                for(std::size_t stage=0;stage<request.size();++stage)if(request[stage].backend==backend) {
                    std::vector<DeviceId> devices;
                    for(const auto id:eligible[backend][owner.rank]) {
                        const std::string_view isa(grouping.device(id).native_isa);
                        if(request[stage].native_isa.empty() || isa.substr(0,isa.find(':'))==request[stage].native_isa)
                            devices.push_back(id);
                    }
                    stage_indices.push_back(stage);candidates.push_back(std::move(devices));
                }
                // Canonical grouping owns the exhaustive disjoint partition,
                // score ordering and immutable UUID ties for both uniform and
                // stage-specific eligibility. Infeasibility is typed; malformed
                // original evidence still throws and cannot become another path.
                std::vector<ObservedPeerStageRequest> declarations;
                for(std::size_t group=0;group<stage_indices.size();++group)
                    declarations.push_back({request[stage_indices[group]].width,candidates[group]});
                auto groups=grouping.select(declarations);
                if(!groups) {fits=false;break;}
                for(std::size_t group=0;group<groups->size();++group) {
                    quality+=grouping.quality((*groups)[group]);
                    candidate.stages[stage_indices[group]]=std::move((*groups)[group]);
                }
            }
            if(!fits)continue;
            std::vector<OriginalIdentity> identity;
            std::set<std::string> selected;
            for(std::size_t stage=0;stage<request.size();++stage) {
                const auto& owner=request[stage].backend==DeviceType::CUDA?cuda:rocm;
                for(const auto id:candidate.stages[stage]) {
                    const auto& gpu=ObservedPeerGrouping(owner).device(id);
                    if(!selected.insert(gpu.uuid).second)
                        throw std::invalid_argument("Cross-vendor stage assignment aliases an original physical GPU");
                    identity.emplace_back(owner.hostname,owner.node_id,owner.cpu.numa_node,owner.rank,
                        gpu.uuid,id.type,id.ordinal);
                }
            }
            // Several complete original owner assignments are legal hardware.
            // Prefer existing peer quality, then immutable original identities
            // in authored stage order, never discovery publication order.
            if(!result || quality>best_quality || (quality==best_quality && identity<best_identity)) {
                best_quality=quality;best_identity=std::move(identity);result=std::move(candidate);
            }
        }
        if(!result)throw std::invalid_argument("No complete original distinct-owner cross-vendor pipeline cohort");
        return *result;
    }
};
} // namespace llaminar2::test
