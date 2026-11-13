/**
 * @file Qwen4ExpObservedPipelineCases.h
 * @brief Original-inventory MI50 EP4+EP4 source candidates without invented GPU addresses.
 *
 * These declarations request architecture and per-stage cardinality. Runtime
 * must bind observed UUID/NUMA ownership before public configuration exists.
 * The existing retained3 source dimension supplies Off/fixed1/fixed3/dynamic;
 * independent native, numerical and source runs must qualify each candidate.
 * A distinct GPU-only GateUpOwnedDownColumns sibling retains fixed-head-sharded
 * VRAM rows and full15 MTP; it never inherits an Apportioned numerical result.
 */
#pragma once
#include "Qwen4ExpSourceAdmissionCases.h"
#include <algorithm>
namespace llaminar2::test::parity::qwen4exp {
/** @return Original discovery and stage policy common to the observed source axes. */
inline ModelParityDefinition qwen4ExpObservedMI50PipelineDefinition() {
    auto definition=qwen4ExpProductionDefinition(Qwen4ExpParityTopology::CPU);
    definition.topology={.test_id="ROCm_MI50_EP4_EP4_2xMPI",
        .kind=ModelParityTopologyKind::RankLocalPipelineParallel,
        .collective=Collective::RCCL,.mpi_ranks=2,
        .pipeline_stage_sizes={4,4},.pipeline_weights={1.0f,1.0f},
        .pipeline_tensor_parallel_collectives={Collective::RCCL,Collective::RCCL},
        .observed_native_pipeline=ModelParityObservedNativePipeline{DeviceType::ROCm,"gfx906"},
        .pipeline_routed_compute_policies={RoutedExpertComputePolicy::Apportioned,RoutedExpertComputePolicy::Apportioned}};
    // Recurring maintenance retains a complete admitted retained3 transaction.
    definition.dynamic_rebalance.economic_movement.device_min_maintenance_period_tokens=4;
    definition.dynamic_rebalance.economic_movement_and_observed_speedup.device_min_maintenance_period_tokens=4;
    return definition;
}

/** @return Normal RAM/mmap x ordinal/random x Static/Dynamic Off seeds. */
inline const std::vector<ModelParityCase>& qwen4ExpObservedMI50PipelineSeeds() {
    static const auto cells=expandModelParityDefinition(qwen4ExpObservedMI50PipelineDefinition());
    return cells;
}
/** @return The existing four-policy retained3 source product, preserving all normal storage/owner/movement axes. */
inline const std::vector<ModelParityCase>& qwen4ExpObservedMI50PipelineSourceCases() {
    static const auto cells=qwen4ExpExpandSourceAdmissionMTP(qwen4ExpObservedMI50PipelineSeeds(),
        Qwen4ExpSourceMTPEnvelope::Retained3);
    return cells;
}
/** @return Q7's four unqualified Off cases with fixed native VRAM ownership. */
inline const std::vector<ModelParityCase>& qwen4ExpObservedMI50FixedVRAMCases() {
    static const auto cells=[] {
        auto definition=qwen4ExpObservedMI50PipelineDefinition();
        definition.inventory_role=ModelParityInventoryRole::NumericalDiagnostic;
        definition.features.ngram_storage={
            ImmutableRowTablePolicy{.storage=ImmutableRowStoragePreference::VRAM,
                                    .distribution=ImmutableRowTableDistribution::FixedHeadSharded}};
        return expandModelParityDefinition(definition);
    }();
    return cells;
}

/** @return Q7's four original storage/movement seeds expanded over retained3 admission. */
inline const std::vector<ModelParityCase>& qwen4ExpObservedMI50FixedVRAMSourceCases() {
    static const auto cells=qwen4ExpExpandSourceAdmissionMTP(qwen4ExpObservedMI50FixedVRAMCases(),
        Qwen4ExpSourceMTPEnvelope::Retained3);
    return cells;
}

/** @return The enabled retained3 functional replay cases, without declaring numerical MTP. */
inline const std::vector<ModelParityCase>& qwen4ExpObservedMI50FixedVRAMExactCases() {
    static const auto cells=[] {
        auto enabled=qwen4ExpObservedMI50FixedVRAMSourceCases();
        std::erase_if(enabled,[](const auto& cell){return !cell.mtpEnabled();});
        return enabled;
    }();
    return cells;
}
/**
 * @return Unbound original MI50 EP4+EP4 request with the explicit GPU hybrid expert policy and full15 cadence.
 * @details Only the routed arithmetic policy changes from the existing observed
 * blueprint. Production discovery still owns GPU order, UUID/NUMA and selected
 * rank membership; no source address or previously observed pass is invented.
 */
inline ModelParityDefinition qwen4ExpObservedMI50GateUpFixedVRAMDefinition()
{
    auto definition=qwen4ExpObservedMI50PipelineDefinition();
    definition.topology.test_id+="_GateUpOwnedDownColumns_Full15";
    definition.topology.pipeline_routed_compute_policies.assign(
        definition.topology.pipeline_stage_sizes.size(),RoutedExpertComputePolicy::GateUpOwnedDownColumns);
    definition.inventory_role=ModelParityInventoryRole::NumericalDiagnostic;
    definition.features.ngram_storage={ImmutableRowTablePolicy{
        .storage=ImmutableRowStoragePreference::VRAM,.distribution=ImmutableRowTableDistribution::FixedHeadSharded}};
    definition.features.dynamic_speedup_witness=ModelParityDynamicSpeedupWitness::Disabled;
    for(auto* movement:{&definition.dynamic_rebalance.economic_movement,
                        &definition.dynamic_rebalance.economic_movement_and_observed_speedup}) {
        movement->device_min_maintenance_period_tokens=kModelParityRequiredMaximumMTPDepth+1;
        movement->device_initial_maintenance_period_tokens=kModelParityRequiredMaximumMTPDepth+1;
    }
    return definition;
}

/**
 * @return Four StaticOrdinal Off/1/3/Dynamic full15 source/numerical/export candidates.
 * @note Source expansion alone declares the original predictor for active MTP;
 * Off retains the inactive paired source for frozen-history corpus closure.
 */
inline const std::vector<ModelParityCase>& qwen4ExpObservedMI50GateUpFixedVRAMCases()
{
    static const auto cases=[] {
        std::vector<ModelParityCase> seeds;
        for(auto cell:expandModelParityDefinition(qwen4ExpObservedMI50GateUpFixedVRAMDefinition()))
            if(cell.expert_overlay==std::optional{ModelParityExpertOverlayPolicy{}}) {
                cell.model.shared_predictor=SharedPredictorGGUFSource{std::string(kPredictorModelPath)};
                seeds.push_back(std::move(cell));
            }
        if(seeds.size()!=1)throw std::logic_error("Observed GateUp PP lost its original StaticOrdinal seed");
        return qwen4ExpExpandSourceAdmissionMTP(seeds,Qwen4ExpSourceMTPEnvelope::Full15);
    }();return cases;
}

/** @return Active byte-exact proofs of the identical GPU hybrid/fixed-head/full15 declarations. */
inline std::vector<ModelParityCase> qwen4ExpObservedMI50GateUpFixedVRAMExactCases()
{
    auto cases=qwen4ExpObservedMI50GateUpFixedVRAMCases();
    std::erase_if(cases,[](const auto& cell){return !cell.mtpEnabled();});
    return cases;
}
} // namespace llaminar2::test::parity::qwen4exp
