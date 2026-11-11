/**
 * @file Qwen4ExpCrossRankPipelineCases.h
 * @brief Real-source two-rank CUDA TP2 to ROCm TP2 inference candidates.
 *
 * These are explicit requested participants, not observed ownership. Original
 * MPI inventory and PMA must prove each local GPU before native preparation.
 * Existing main20/source-only compositions retain their original identities.
 * Declaration supplies no inference, numerical or multi-stage MTP certificate.
 */
#pragma once
#include "Qwen4ExpSourceAdmissionCases.h"
#include "planning/RankHardwareOwnership.h"
#include <set>

namespace llaminar2::test::parity::qwen4exp
{
/**
 * @return Original full48/final-logit diagnostic on two real local TP domains.
 * @details CUDA0/1 belong to requested original rank0; ROCm1/2 to rank1.
 * Physical NUMA identity is authenticated from original inventory, never
 * calculated from those ranks or GPU ordinals. Equal24-layer stages are an
 * explicit requested partition. Real capacity admission may reject it.
 */
inline const std::vector<ModelParityCase>& qwen4ExpCrossRankPipelineOrdinaryCases()
{
    static const auto cases=[] {
        auto definition=qwen4ExpProductionDefinition(Qwen4ExpParityTopology::CUDAROCmCPU);
        definition.topology={
            .test_id="CUDA01_ROCm12_TP2_PP2_World2",
            .kind=ModelParityTopologyKind::NodePipelineParallel,
            .participants={{GlobalDeviceAddress::cuda(0),0},{GlobalDeviceAddress::cuda(1),0},
                           {GlobalDeviceAddress::rocm(1),1},{GlobalDeviceAddress::rocm(2),1}},
            .collective=Collective::HETEROGENEOUS,.mpi_ranks=2,
            .pipeline_stage_sizes={2,2},.pipeline_weights={1.0f,1.0f},
            .pipeline_tensor_parallel_collectives={Collective::NCCL,Collective::RCCL}};
        definition.features.ngram_storage={{.storage=ImmutableRowStoragePreference::RAM}};
        definition.features.mtp=ModelParityAxisProfile::Disabled;
        definition.model.maximum_mtp_draft_depth=0;
        definition.model.shared_predictor.reset();
        return expandModelParityDefinition(definition);
    }();
    return cases;
}
/**
 * @return Same real-source placement crossed with fixed1/fixed3/dynamic exact proof.
 * @details Reuses the existing retained-three policy projection and original
 * serial/grouped/reset oracle. These are native exact-proof candidates, not
 * reference-scored MTP evidence. Model-free multi-stage execution is qualified
 * independently; these real-source placements still require their own exact
 * source, numerical, reset/prefix and performance evidence.
 */
inline const std::vector<ModelParityCase>& qwen4ExpCrossRankPipelineMTPExactCases()
{
    static const auto cases=[] {
        auto result=qwen4ExpExpandSourceAdmissionMTP(qwen4ExpCrossRankPipelineOrdinaryCases(),
            Qwen4ExpSourceMTPEnvelope::Retained3);
        std::erase_if(result,[](const auto& cell){return cell.mtp==ModelParityMTP::Off;});
        return result;
    }();
    return cases;
}
/**
 * @brief Authenticate the requested physical cohort before preparing any worker.
 * @param cell Original typed NodePipelineParallel request.
 * @param inventory Original complete MPI publication, without a new probe.
 * @throws std::invalid_argument For absent/foreign ownership, duplicate physical
 * identity, remote physical nodes or a mismatched exact original world.
 */
inline void requireQwen4ExpCrossRankPipelineOwners(const ModelParityCase& cell,
    const ClusterInventory& inventory)
{
    if(cell.topology.kind!=ModelParityTopologyKind::NodePipelineParallel ||
       cell.topology.mpi_ranks!=2 || inventory.ranks.size()!=2)
        throw std::invalid_argument("QFN cross-rank pipeline requires its exact original two-rank world");
    std::set<std::string> uuids;
    std::optional<int> node;
    for(const auto& participant:cell.topology.participants)
    {
        if(!participant.world_rank)
            throw std::invalid_argument("QFN cross-rank pipeline lost explicit original ownership");
        const auto rank=std::find_if(inventory.ranks.begin(),inventory.ranks.end(),
            [&](const auto& observed){return observed.rank==*participant.world_rank;});
        if(rank==inventory.ranks.end() || rank->node_id<0 || (node && *node!=rank->node_id))
            throw std::invalid_argument("QFN cross-rank pipeline requires one observed physical node");
        node=rank->node_id;
        const DeviceId device{participant.address.device_type,participant.address.device_ordinal};
        requireRankLocalGPUOwnership(*rank,device,"QFN original cross-rank pipeline");
        const auto gpu=std::find_if(rank->gpus.begin(),rank->gpus.end(),[&](const auto& value){
            return value.type==device.type && value.local_device_id==device.ordinal;});
        if(gpu->uuid.empty() || !uuids.insert(gpu->uuid).second)
            throw std::invalid_argument("QFN cross-rank pipeline requires distinct observed physical GPU UUIDs");
    }
}
} // namespace llaminar2::test::parity::qwen4exp
