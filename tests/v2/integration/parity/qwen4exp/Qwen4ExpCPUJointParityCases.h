/**
 * @file Qwen4ExpCPUJointParityCases.h
 * @brief Exact original-source CPU joint full-stage diagnostic case.
 *
 * The source-admission seed supplies only its original physical topology.
 * Numerical execution retains the original model/predictor/arithmetic and all
 * checkpoint policies; no source-only result or numerical badge is inherited.
 */
#pragma once
#include "Qwen4ExpSourceAdmissionCases.h"
namespace llaminar2::test::parity::qwen4exp {
/** @return One Static RAM Off/capacity0 numerical request on the original one-rank/socket topology. */
inline const std::vector<ModelParityCase>& qwen4ExpCPUJointWorld2NumericalCases()
{
    static const auto cases=[] {
        auto definition=qwen4ExpProductionDefinition(Qwen4ExpParityTopology::ROCmCPU);
        definition.topology=qwen4ExpCPUJointWorld2SourceAdmissionCases().front().topology;
        definition.features.mtp=ModelParityAxisProfile::Disabled;
        definition.features.dynamic_speedup_witness=ModelParityDynamicSpeedupWitness::Disabled;
        definition.features.ngram_storage={{.storage=ImmutableRowStoragePreference::RAM}};
        definition.model.maximum_mtp_draft_depth=0;
        // Preserve the frozen corpus original source closure without preparing active MTP.
        definition.model.shared_predictor=SharedPredictorGGUFSource{std::string(kPredictorModelPath)};
        std::vector<ModelParityCase> result;
        for(auto cell:expandModelParityDefinition(definition))
            if(cell.expert_overlay==std::optional{ModelParityExpertOverlayPolicy{}})result.push_back(std::move(cell));
        if(result.size()!=1 || result.front().mtp!=ModelParityMTP::Off || result.front().retained_mtp_draft_capacity!=0)
            throw std::logic_error("CPU joint numerical request lost its original Static RAM Off identity");
        return result;
    }();
    return cases;
}
/**
 * @return Explicit fixed1/fixed3/dynamic native serial/grouped proof requests.
 * @note These retain the original world2 socket/source topology and canonical
 * exact proof. They add no reference-sidecar numerical acceptance or pass badge.
 */
inline const std::vector<ModelParityCase>& qwen4ExpCPUJointWorld2MTPExactCases()
{
    static const auto cases=[] {
        std::vector<ModelParityCase> result;
        for(auto cell:qwen4ExpCPUJointWorld2MTPSourceAdmissionCases()) {
            if(cell.mtp==ModelParityMTP::Off)continue;
            cell.inventory_role=ModelParityInventoryRole::NumericalDiagnostic;
            cell.collective_evidence_source=ParityCollectiveEvidenceSource::PostCollectiveSnapshot;
            cell.thresholds.allreduce_stages.clear();
            result.push_back(std::move(cell));
        }
        return result;
    }();
    return cases;
}

}
