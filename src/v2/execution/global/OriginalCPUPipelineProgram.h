/**
 * @file OriginalCPUPipelineProgram.h
 * @brief Stage-scoped CPU expert collective declaration over original MPI/NUMA owners.
 *
 * Planning and native setup consume the same checked original program. The
 * value is immutable metadata, not a communicator, hardware cache or physical
 * admission. Every parent rank visits declarations in original stage order;
 * excluded ranks retain their genuine process role and enter the same split.
 */
#pragma once
#include "collective/NativeCPUCollectiveProgram.h"
#include "execution/global_pp/GlobalPPTopology.h"
#include "execution/factory/FactoryPPStageConfig.h"
#include "planning/RankHardwareOwnership.h"
#include "config/OrchestrationConfig.h"
#include "config/BackendSelector.h"
#include <limits>
#include <optional>
namespace llaminar2 {
/** @brief Checked original stage and native program, with one canonical identity. */
class OriginalCPUPipelineProgram final {
public:
    /**
     * @brief Declare an original routed CPU stage after topology compilation.
     * @param stage Original global stage and complete ordered per-rank endpoints.
     * @param inventory Exact parent rank publication; no new discovery occurs.
     * @return Same immutable native program used by BOM admission and communicator creation.
     * @throws std::invalid_argument for foreign ranks, NUMA ownership, duplicates or non-CPU membership.
     */
    static OriginalCPUPipelineProgram checked(const GlobalPPStageSpec& stage,const ClusterInventory& inventory) {
        if(!stage.is_global_tp || stage.stage_id<0 || stage.first_layer<0 || stage.last_layer<stage.first_layer ||
            stage.last_layer==std::numeric_limits<int>::max() || stage.per_rank_devices.size()<2 ||
            stage.participating_ranks.size()!=stage.per_rank_devices.size())
            throw std::invalid_argument("CPU pipeline program requires its complete original global TP stage");
        std::vector<NativeCPUCollectiveProgram::Participant> members;
        int node=-1;
        for(std::size_t i=0;i<stage.participating_ranks.size();++i) {
            const int rank=stage.participating_ranks[i];const auto& address=stage.per_rank_devices[i];
            if(rank<0 || rank>=inventory.world_size || std::size_t(rank)>=inventory.ranks.size())
                throw std::invalid_argument("CPU pipeline program has a foreign original MPI rank");
            const auto& owner=inventory.ranks[rank];
            if(owner.rank!=rank || !address.isCPU() || address.hostname!=owner.hostname ||
                owner.cpu.numa_node<0 || address.numa_node!=owner.cpu.numa_node ||
                (node>=0 && owner.node_id!=node))
                throw std::invalid_argument("CPU pipeline program changed original node-local NUMA ownership");
            node=owner.node_id;
            for(const auto& prior:members) if(prior.world_rank==rank ||
                (prior.address.hostname==address.hostname && prior.address.numa_node==address.numa_node))
                throw std::invalid_argument("CPU pipeline program duplicates an original physical/process owner");
            members.push_back({rank,address});
        }
        const auto backend=BackendSelector::resolve(stage.backend,std::vector<DeviceId>(members.size(),DeviceId::cpu()));
        return OriginalCPUPipelineProgram(stage.stage_id,
            {.first_layer=stage.first_layer,.last_layer=stage.last_layer+1,.has_embedding=stage.has_embedding,
             .has_lm_head=stage.has_lm_head,.generation_domain=DeviceGenerationExecutionDomain::HostedBoundary},
            NativeCPUCollectiveProgram::expertShardRoutePartials(stage.stage_id,backend,std::move(members)));
    }
    /** @return Original global stage key, never a rank or NUMA inference. */
    int stageId() const noexcept {return stage_id_;}
    /** @return Original closed native operation family and ordered membership. */
    const NativeCPUCollectiveProgram& nativeProgram() const & noexcept {return native_;}
    const NativeCPUCollectiveProgram& nativeProgram() const && = delete;
    /** @return Original model interval and component roles priced for this stage. */
    const FactoryPPStageConfig& scope() const noexcept {return scope_;}
private:
    /** @brief Only checked original topology creates a CPU stage program. */
    OriginalCPUPipelineProgram(int id,FactoryPPStageConfig scope,NativeCPUCollectiveProgram native)
        :stage_id_(id),scope_(scope),native_(std::move(native)){}
    int stage_id_;
    FactoryPPStageConfig scope_;
    NativeCPUCollectiveProgram native_;
};
/**
 * @brief Resolve every authored CPU shard-stage program on the original parent namespace.
 * @param config Original stage policies; per-domain values override inherited defaults.
 * @param topology Canonical compiled stage membership, in semantic stage order.
 * @param inventory Same immutable original rank/NUMA publication.
 * @return Stage-ID sorted programs; each routed CPU TP/EP stage retains one admitted original context.
 */
inline std::vector<OriginalCPUPipelineProgram> originalCPUPipelinePrograms(const OrchestrationConfig& config,
    const GlobalPPTopology& topology,const ClusterInventory& inventory) {
    std::vector<OriginalCPUPipelineProgram> result;
    for(const auto& stage:topology.stages) {
        if(!stage.is_global_tp || stage.per_rank_devices.empty() ||
            !std::all_of(stage.per_rank_devices.begin(),stage.per_rank_devices.end(),[](const auto& p){return p.isCPU();})) continue;
        auto compute=config.routed_expert_compute_policy;
        for(const auto& domain:config.domain_definitions) if(domain.name==stage.domain_name &&
            domain.routed_compute_policy!=RoutedExpertComputePolicy::Unspecified) compute=domain.routed_compute_policy;
        if(compute!=RoutedExpertComputePolicy::Automatic && compute!=RoutedExpertComputePolicy::Apportioned &&
            compute!=RoutedExpertComputePolicy::TensorSharded)
            throw std::invalid_argument("CPU pipeline program has unsupported expert ownership");
        result.push_back(OriginalCPUPipelineProgram::checked(stage,inventory));
    }
    std::sort(result.begin(),result.end(),[](const auto& a,const auto& b){return a.stageId()<b.stageId();});
    return result;
}
}
