/**
 * @file Qwen4ExpCPUJointSocketEvidence.h
 * @brief Original-plan physical socket qualification for source-only CPU expert fragments.
 *
 * Expected roles come only from the canonical placement plan. Native validation
 * consumes the SAME original MPI inventory later retained by the Runner, never
 * another hardware discovery or hostname-based locality inference. Socket IDs
 * have meaning only inside their original physical node namespace.
 */
#pragma once
#include "../ModelParityDefinition.h"
#include "execution/moe/MoEExpertOwnerMap.h"
#include "execution/mpi_orchestration/DeviceInventory.h"
#include "fixtures/collective/CPUExpertDomainSocketMembership.h"
#include <algorithm>
#include <map>
#include <set>
#include <stdexcept>
#include <vector>
namespace llaminar2::test::parity::qwen4exp {
/**
 * @brief Project the actual referenced CPU-joint domain, without ownership or hardware fabrication.
 * @param cell Original canonical source request.
 * @return Original CPU addresses/ranks, or empty for ordinary source cases.
 * @throws std::invalid_argument For multiple domains, unresolved ranks or incomplete socket-per-rank intent.
 */
inline std::vector<ModelParityParticipant> qwen4ExpCPUJointSocketMembers(const ModelParityCase& cell) {
    const auto plan=cell.topology.expert_overlay_plan;
    if(!plan || !MoEExpertOwnerMap::usesCPUJointFragments(*plan))return {};
    const auto participants=MoEExpertOwnerMap::declaredParticipants(*plan,{.reject_tensor_sharded_domains=false});
    std::vector<ModelParityParticipant> result;std::set<int> ranks,nodes;std::set<std::string> domains;
    for(const auto& participant:participants) {
        const auto domain=std::find_if(plan->domains.begin(),plan->domains.end(),
            [&](const auto& candidate){return candidate.name==participant.domain_name;});
        if(domain==plan->domains.end())throw std::invalid_argument("CPU source participant has no original domain");
        if(!participant.address.isCPU() || domain->routed_compute_policy!=RoutedExpertComputePolicy::TensorSharded ||
           domain->participants.size()<=1)continue;
        if(domain->scope!=ExecutionDomainScope::NODE_LOCAL || !participant.world_rank_known ||
           participant.world_rank<0 || participant.world_rank>=cell.topology.mpi_ranks ||
           participant.address.device_ordinal!=0 || !participant.address.hasValidNuma() ||
           !ranks.insert(participant.world_rank).second || !nodes.insert(participant.address.numa_node).second)
            throw std::invalid_argument("CPU joint source requires one original rank per distinct declared NUMA endpoint");
        const auto physical=std::count_if(cell.topology.participants.begin(),cell.topology.participants.end(),
            [&](const auto& candidate){return candidate.world_rank==participant.world_rank && candidate.address.isCPU() &&
                candidate.address.device_ordinal==0 && candidate.address.numa_node==participant.address.numa_node;});
        if(physical!=1)throw std::invalid_argument("CPU joint source socket member is absent or duplicated in original physical roles");
        domains.insert(domain->name);result.push_back({participant.address,participant.world_rank});
    }
    if(domains.size()!=1 || cell.topology.mpi_ranks<2 || cell.topology.mpi_ranks>8 ||
       result.size()!=size_t(cell.topology.mpi_ranks))
        throw std::invalid_argument("CPU joint source requires its one complete original socket domain");
    return result;
}
/**
 * @brief Classify an original declared participant without constructing expert ownership rows.
 * @param plan Original frozen/declared source plan.
 * @param participant Original descriptor from declaredParticipants().
 * @return Whether this descriptor names a multi-member CPU tensor fragment.
 * @throws std::invalid_argument For a CPU descriptor foreign to the original domain inventory.
 */
inline bool qwen4ExpOriginalCPUJointParticipant(const MoERoutedExpertPlacementPlan& plan,
    const MoEExpertOwnerParticipant& participant) {
    if(!participant.address.isCPU())return false;
    const auto domain=std::find_if(plan.domains.begin(),plan.domains.end(),
        [&](const auto& candidate){return candidate.name==participant.domain_name;});
    if(domain==plan.domains.end())throw std::invalid_argument("CPU source descriptor is foreign to its original domains");
    const bool joint=domain->routed_compute_policy==RoutedExpertComputePolicy::TensorSharded && domain->participants.size()>1;
    if(joint && (!participant.world_rank_known || participant.world_rank<0))
        throw std::invalid_argument("CPU source fragment has no original resolved world rank");
    return joint;
}
/**
 * @brief Count the exact source-layer observations owed by original local fragment participants.
 * @param plan Original source placement interval; no ownership or quota reconstruction occurs.
 * @param participants Same original declared descriptor set used by the observer.
 * @param rank Original world rank, including a rank with no CPU fragment.
 * @return Local joint-member count times original placement-row count, possibly zero.
 */
inline size_t qwen4ExpCPUJointSourceExpectedLayerObservations(const MoERoutedExpertPlacementPlan& plan,
    std::span<const MoEExpertOwnerParticipant> participants,int rank) {
    if(rank<0)throw std::invalid_argument("CPU source observation has no original world rank");
    const auto local=std::count_if(participants.begin(),participants.end(),[&](const auto& participant){
        return qwen4ExpOriginalCPUJointParticipant(plan,participant) && participant.world_rank==rank;});
    return static_cast<size_t>(local)*plan.placements.size();
}
/**
 * @brief Project one unambiguous physical package from frozen CPU socket facts.
 * @param sockets Original local-node inventory, not inferred from NUMA numbering.
 * @param numa Required original NUMA identity.
 * @return Exact observed package ID, or absent for missing/ambiguous evidence.
 */
inline std::optional<int> qwen4ExpOriginalSocketForNUMA(std::span<const CPUSocketInfo> sockets,int numa) noexcept {
    std::optional<int> result;
    if(numa<0)return {};
    for(const auto& socket:sockets)if(socket.numa_node==numa) {
        if(socket.socket_id<0 || result)return {};
        result=socket.socket_id;
    }
    return result;
}
/**
 * @brief Validate original bound rank/NUMA/socket observations before model load.
 * @param cell Original canonical source request.
 * @param inventory Already-published inventory from the SAME retained MPI context used by the Runner.
 * @return Validated original endpoint observations, empty for ordinary source cases.
 * @throws std::invalid_argument For incomplete, foreign-node, aliased, stale or incorrectly bound observations.
 * @note This performs no discovery, native calls, allocation decision or communicator construction.
 */
inline std::vector<test::CPUExpertSocketEndpoint> requireOriginalCPUJointSocketRanks(
    const ModelParityCase& cell,const ClusterInventory& inventory) {
    const auto required=qwen4ExpCPUJointSocketMembers(cell);if(required.empty())return {};
    if(inventory.ranks.size()!=required.size())throw std::invalid_argument("CPU joint source has incomplete original rank inventory");
    std::set<int> observed_ranks;std::optional<int> original_node;
    std::optional<std::map<int,int>> original_sockets;
    std::vector<test::CPUExpertSocketEndpoint> endpoints;
    for(const auto& member:required) {
        const auto rank=std::find_if(inventory.ranks.begin(),inventory.ranks.end(),
            [&](const auto& candidate){return candidate.rank==*member.world_rank;});
        if(rank==inventory.ranks.end() || rank->node_id<0 || rank->cpu.type!=DeviceType::CPU ||
           !observed_ranks.insert(rank->rank).second ||
           rank->cpu.numa_node!=member.address.numa_node ||
           (original_node && *original_node!=rank->node_id))
            throw std::invalid_argument("CPU joint source rank is not bound to its original physical node/NUMA endpoint");
        original_node=rank->node_id;
        std::map<int,int> sockets;std::set<int> socket_ids;
        for(const auto& socket:rank->cpu_socket_info) {
            if(socket.socket_id<0 || socket.numa_node<0 || !sockets.emplace(socket.numa_node,socket.socket_id).second ||
               !socket_ids.insert(socket.socket_id).second)
                throw std::invalid_argument("CPU joint source has missing or duplicate original socket evidence");
        }
        if(original_sockets && *original_sockets!=sockets)
            throw std::invalid_argument("CPU joint source has inconsistent socket facts inside the same original node");
        original_sockets=sockets;
        const auto socket=sockets.find(rank->cpu.numa_node);
        if(socket==sockets.end())throw std::invalid_argument("CPU joint source bound NUMA has no original physical socket");
        endpoints.push_back({rank->rank,rank->cpu.numa_node,socket->second});
    }
    if(!test::validCPUExpertSocketMembership(endpoints,cell.topology.mpi_ranks))
        throw std::invalid_argument("CPU joint source requires one original MPI rank per distinct physical socket");
    return endpoints;
}
} // namespace llaminar2::test::parity::qwen4exp
