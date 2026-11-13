/** @file RecordedHostServicePlanning.h
 * @brief Recorded role budgets for actual model-free capacity planning, with explicit physical topology.
 *
 * Existing multi-rank fixtures retain their full authored ClusterInventory.
 * Legacy single-rank observations are explicitly placed together on node0;
 * no remote rank or native resource observation is invented. The actual planner
 * still validates every field and executes every production BOM/quota path.
 */
#pragma once
#include "../../utils/PlanningHostServiceFixture.h"
#include "execution/moe/MoEOverlayLocalCapacityPlanner.h"
#include "execution/mpi_orchestration/DeviceInventory.h"
namespace llaminar2::test {
/** @return Sealed recorded CPU role inputs; no installed native qualification. */
inline RetainedHostServiceBudgetProjection recordedHostServiceBudget(){
    return planningHostServiceProjection();
}
/** @return Explicit original rank0/node0 fixture inventory, never a guessed remote topology. */
inline ClusterInventory recordedSingleRankCluster(const RankInventory& rank){
    if(rank.rank!=0)throw std::invalid_argument("Metadata fixture must author full inventory for a nonzero/remote rank");
    ClusterInventory cluster;cluster.world_size=1;cluster.node_count=1;cluster.ranks={rank};
    cluster.ranks.front().node_id=0;
    cluster.nodes={{.node_id=0,.hostname="recorded-local-node",.ranks={0}}};return cluster;
}
/** @brief Add only recorded CPU role inputs; execute the actual capacity planner synchronously. */
inline MoEOverlayLocalCapacityPlannerResult planWithRecordedHostServices(MoEOverlayLocalCapacityPlannerInput input){
    input.retained_host_services=recordedHostServiceBudget();
    ClusterInventory local;
    if(!input.cluster_inventory&&input.rank_inventory){local=recordedSingleRankCluster(*input.rank_inventory);input.cluster_inventory=&local;}
    return MoEOverlayLocalCapacityPlanner::plan(input);
}
/** @brief Keep actual shared pipeline admission while supplying each authored stage's recorded role inputs. */
inline std::vector<MoEOverlayLocalCapacityPlannerResult> planPipelineWithRecordedHostServices(
    const RankExecutionPlan& parent,std::span<const MoEOverlayLocalCapacityPlannerInput> original){
    std::vector<MoEOverlayLocalCapacityPlannerInput> stages(original.begin(),original.end());
    std::vector<ClusterInventory> locals(stages.size());
    const auto projection=recordedHostServiceBudget();
    for(std::size_t i=0;i<stages.size();++i){
        stages[i].retained_host_services=projection;
        if(!stages[i].cluster_inventory&&stages[i].rank_inventory){locals[i]=recordedSingleRankCluster(*stages[i].rank_inventory);stages[i].cluster_inventory=&locals[i];}
    }
    return MoEOverlayLocalCapacityPlanner::planPipeline(parent,stages);
}
}
