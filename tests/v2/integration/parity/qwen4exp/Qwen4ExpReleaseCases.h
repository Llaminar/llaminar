/**
 * @file Qwen4ExpReleaseCases.h
 * @brief Three QFN release controls through the existing full-stage matrix.
 *
 * RAM owns ngram rows independently of native GPU expert ownership. Pure GPU
 * TP uses the existing all-GPU projection blueprint without a CPU expert tier.
 * The heterogeneous path has explicit TP domains and pipeline boundaries.
 * Every case must prove real source admission, native capture and all stages.
 */
#pragma once
#include "Qwen4ExpFullROCmTP4ParityCases.h"
#include "../ModelParityReleaseTopologies.h"

namespace llaminar2::test::parity::qwen4exp
{
/** @return Exactly three MTP-off/RAM controls, with no storage or placement product. */
inline const std::vector<ModelParityCase>& qwen4ExpReleaseCases()
{
    static const auto cases=[] {
        std::vector<ModelParityCase> result;
        for (auto topology : modelParityReleaseTopologies())
        {
            auto definition=qwen4ExpProductionDefinition(Qwen4ExpParityTopology::CPU);
            if (topology.kind==ModelParityTopologyKind::RankLocalTensorParallel)
            {
                auto plan=std::make_shared<MoERoutedExpertPlacementPlan>(
                    *qwen4ExpFullROCmTP4Topology().expert_overlay_plan);
                auto &domain=plan->domains.front();
                domain.name="qfn_release_gpu_tp";
                domain.backend=topology.collective==Collective::NCCL
                    ? CollectiveBackendType::NCCL : CollectiveBackendType::RCCL;
                domain.participants.clear();
                for(const auto &participant:topology.participants)
                    domain.participants.push_back(participant.address);
                domain.weights.assign(domain.participants.size(),1.0f/domain.participants.size());
                plan->continuation_domain=plan->base_model_domain=plan->shared_expert_domain=domain.name;
                plan->continuation_domain_spec.domain=domain.name;
                plan->dense_domains={domain.toExecutionDomainDefinition()};
                plan->routed_tiers.front().domain=domain.name;
                if(!validateMoERoutedExpertPlacementPlan(*plan).ok())
                    throw std::invalid_argument("Invalid release GPU expert ownership");
                topology.expert_overlay_plan=std::move(plan);
            }
            definition.topology=std::move(topology);
            definition.features.ngram_storage={{.storage=ImmutableRowStoragePreference::RAM}};
            ModelParityE2ESelection selected{.ngram_table_policy=definition.features.ngram_storage.front()};
            if(definition.topology.isExpertOverlay())
            {
                selected.owner_order=RoutedExpertOwnerOrder::Ordinal;
                selected.movement=ModelParityExpertMovement::Static;
            }
            definition.e2e_certifiable={selected};
            // The canonical expander retains numerical policy siblings. Only
            // the exact source-selected control enters the bounded release family.
            for(auto cell:expandModelParityDefinition(definition))
                if(cell.e2e_certification)result.push_back(std::move(cell));
        }
        if(result.size()!=3)throw std::logic_error("QFN release inventory requires exactly three controls");
        return result;
    }();
    return cases;
}
} // namespace llaminar2::test::parity::qwen4exp
