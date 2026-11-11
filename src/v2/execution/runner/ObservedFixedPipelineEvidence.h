/**
 * @file ObservedFixedPipelineEvidence.h
 * @brief Passive original selected-rank and native fixed-pipeline role evidence.
 *
 * The running application retains the discovery and selected MPI namespaces.
 * This observer consumes their immutable inventory plus the actual runner's
 * admitted stage and storage owners. It opens no device, communicator or table.
 */
#pragma once

#include "execution/runner/IOrchestrationRunner.h"
#include "execution/qwen4exp/FixedHeadPipelineTerminalEvidence.h"
#include "interfaces/IModelContext.h"
#include "loaders/ImmutableFixedHeadPipelineProjection.h"
#include "loaders/ResidentImmutableRowTable.h"
#include "loaders/ResolvedImmutableRowTablePlacement.h"
#include "planning/ExecutionRankMembership.h"
#include "planning/ObservedPeerGrouping.h"
#include "planning/RankHardwareOwnership.h"
#include "utils/MPIContext.h"
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace llaminar2 {

/**
 * @brief Identify an authored selected fixed-head GPU pipeline observation.
 * @param config Original parsed configuration, before or after runner admission.
 * @return True only for a fixed VRAM pipeline with explicit original discovery selection.
 * @throws std::invalid_argument For inconsistent selected execution geometry.
 */
inline bool isObservedFixedNativePipeline(const OrchestrationConfig& config) {
    if (!config.execution_rank_selection || config.mpi_procs <= 1 ||
        config.pp_stage_definitions.empty() || !config.ngram_table_policy.isFixedHeadSharded())
        return false;
    if (config.planning_mode != OrchestrationPlanningMode::Apply || !config.hostfile.empty() ||
        config.execution_rank_selection->size() != 1 || config.pp_stage_definitions.size() < 2 ||
        config.ngram_table_policy.storage != ImmutableRowStoragePreference::VRAM ||
        config.execution_rank_selection->discoveryRanks().front() >= config.mpi_procs)
        throw std::invalid_argument("Observed fixed native pipeline requires one original selected GPU owner and complete native stages");
    return true;
}

/**
 * @brief Observe exact selected discovery hardware and the admitted source/PLE roster.
 * @param runner The active original model runner after complete preparation.
 * @param execution The same selected MPI context retained by the frontend.
 * @return Passive ordered physical/role evidence for numerical or Release artifacts.
 * @throws std::invalid_argument When a stage, original resource or PLE role changes.
 */
inline nlohmann::json observeFixedNativePipelineBinding(const IOrchestrationRunner& runner,
    const IMPIContext& execution)
{
    const auto& config=runner.config();
    const auto* selected=dynamic_cast<const MPIContext*>(&execution);
    const auto* membership=selected?selected->executionMembership():nullptr;
    if(!isObservedFixedNativePipeline(config) || !membership ||
        membership->selection()!=*config.execution_rank_selection ||
        membership->discoverySize()!=config.mpi_procs || execution.world_size()!=1 || execution.rank()!=0)
        throw std::invalid_argument("Observed fixed native pipeline lost original selected MPI membership");
    const auto* discovery=selected->originalDiscoveryContext();
    const auto inventory=discovery?discovery->clusterInventory():nullptr;
    const int original_rank=membership->discoveryRanks().front();
    if(!inventory || discovery->world_size()!=membership->discoverySize() || discovery->rank()!=original_rank)
        throw std::invalid_argument("Observed fixed native pipeline cannot authenticate original discovery ownership");
    const auto& original=inventory->ranks.at(original_rank);
    const auto& projected=membership->inventory().ranks.at(execution.rank());
    if(original.hostname!=projected.hostname || original.cpu.numa_node!=projected.cpu.numa_node)
        throw std::invalid_argument("Observed fixed native pipeline changed its physical host or NUMA owner");

    const auto& plan=runner.executionPlan();
    const auto* model=runner.modelContextForDiagnostics();
    const auto placement=model?model->resolvedImmutableRowTablePlacement():nullptr;
    const auto* fixed=placement?placement->fixedHeadPipeline():nullptr;
    if(!fixed || model->hostImmutableRowProvider() || plan.rank!=execution.rank() || !plan.usesLocalPP() ||
        plan.local_pp_devices.size()!=config.pp_stage_definitions.size() ||
        plan.local_pp_stage_tp_info.size()!=plan.local_pp_devices.size())
        throw std::invalid_argument("Observed fixed native pipeline lost actual admitted native stages or PLE placement");
    const ObservedPeerGrouping source_devices(original), selected_devices(projected);
    nlohmann::json result={{"discovery_size",membership->discoverySize()},
        {"selected_discovery_ranks",membership->discoveryRanks()},{"original_owner",original_rank},
        {"original_hostname",original.hostname},{"original_numa_node",original.cpu.numa_node},
        {"execution_rank",execution.rank()},{"stage_sizes",nlohmann::json::array()},
        {"participants",nlohmann::json::array()},{"fixed_owners",nlohmann::json::array()},
        {"ple_consumers",nlohmann::json::array()},{"ple_stage",fixed->consumerStage()}};
    std::size_t population=0;
    for(std::size_t stage=0;stage<plan.local_pp_devices.size();++stage) {
        const auto& group=plan.local_pp_stage_tp_info[stage];
        if(group.devices.empty() || group.devices.front()!=plan.local_pp_devices[stage] ||
           group.domain_name!=config.pp_stage_definitions[stage].domain_name)
            throw std::invalid_argument("Observed fixed native pipeline changed an original stage roster");
        result["stage_sizes"].push_back(group.devices.size());
        population+=group.devices.size();
        for(std::size_t member=0;member<group.devices.size();++member) {
            const auto& address=group.devices[member];
            const auto device=address.toLocalDeviceId();
            requireRankLocalGPUOwnership(original,device,"Observed fixed native pipeline");
            const auto& before=source_devices.device(device);
            const auto& after=selected_devices.device(device);
            const auto& role=fixed->owner({execution.rank(),device});
            const auto table=model->residentImmutableRowTable(device);
            if(before.uuid!=after.uuid || before.native_isa!=after.native_isa || before.numa_node!=after.numa_node ||
               before.pcie_upstream_bdfs!=after.pcie_upstream_bdfs || address.numa_node!=before.numa_node ||
               role.stage_index!=static_cast<int>(stage) || role.participant_index!=static_cast<int>(member) ||
               !table || table->sourceRange()!=placement->fixedHeadSourceRangeFor(role.allocator))
                throw std::invalid_argument("Observed fixed native pipeline changed original hardware or native range role");
            result["participants"].push_back({{"stage",stage},{"backend",device.is_rocm()?"rocm":"cuda"},
                {"ordinal",device.ordinal},{"uuid",before.uuid},{"native_isa",before.native_isa},
                {"numa_node",before.numa_node},{"pcie_upstream_bdfs",before.pcie_upstream_bdfs}});
        }
    }
    if(population!=fixed->owners().size())
        throw std::invalid_argument("Observed fixed native pipeline has a foreign storage owner");
    for(const auto& owner:fixed->owners()) {
        const auto range=placement->fixedHeadSourceRangeFor(owner.allocator);
        result["fixed_owners"].push_back({{"device",owner.allocator.device.to_string()},
            {"stage",owner.stage_index},{"participant",owner.participant_index},{"consumer",owner.consumer_index},
            {"first_head",owner.first_head},{"heads",owner.head_count},{"native_bytes",range.nativeBytes()}});
    }
    for(const auto& consumer:fixed->consumers())
        result["ple_consumers"].push_back(consumer.device.to_string());
    return result;
}

/**
 * @brief Export the existing generation terminal receipt totals for the original fixed owners.
 * @param runner Same prepared runner after all observed generation or force work.
 * @param expected_rows Independent exact input rows for a fixed numerical history.
 * @return Validated useful native traffic and completed receipt totals per device.
 * @throws std::invalid_argument For absent, failed or mis-sized original native work.
 */
nlohmann::json observeFixedNativePipelineTerminal(const IOrchestrationRunner& runner,
    std::optional<std::uint64_t> expected_rows = {});

} // namespace llaminar2
