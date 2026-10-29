/**
 * @file Qwen4ExpProductionParityCases.h
 * @brief Canonical Flash Next topology/storage matrix and exact reference identity.
 *
 * The ordinary decoder is independently authenticated from its four GGUF
 * shards. Each single-vendor GPU continuation uses ExpertOverlay with CPU
 * expert capacity. The mixed-vendor case is an explicit CUDA/ROCm/CPU pipeline:
 * each device owns its graph and complete local expert work. RAM and bounded
 * mmap are independent cells; only overlay cases expand movement/owner axes.
 * Additional CPU-NUMA and composed TP/PP blueprints are consumed by the plain
 * source-admission fixture only; they do not extend the numerical main20 list.
 */
#pragma once
#include "Qwen4ExpModelParityDefinitions.h"
#include "execution/moe/MoERoutedExpertPlacementPlan.h"
#include <array>
#include <iterator>

namespace llaminar2::test::parity::qwen4exp
{
/** @brief Requested physical backend composition, independent of storage choice. */
enum class Qwen4ExpParityTopology
{
    CPU, CUDACPU, ROCmCPU, CUDAROCmCPU,
    CPU2TP, CPU2PP, ROCm4TP2PP2, CUDA2CPU2TPPP, ROCm2CPU2TPPP,
    CUDA2ROCm2CPU2TPPP,
};
/** @brief One exact participant domain; quotas remain the production memory planner's responsibility. */
inline RoutedExpertDomain qwen4ExpParityDomain(const char *name, GlobalDeviceAddress device,
                                              CollectiveBackendType backend)
{
    RoutedExpertDomain domain;
    domain.name = name;
    domain.scope = ExecutionDomainScope::SINGLE;
    domain.backend = backend;
    domain.owner_rank = 0;
    domain.participants = {std::move(device)};
    domain.routed_compute_policy = RoutedExpertComputePolicy::Apportioned;
    domain.routed_phase_policy = RoutedExpertPhasePolicy::Uniform;
    domain.routed_decode_assignment_policy = RoutedExpertAssignmentPolicy::StaticOwner;
    domain.routed_prefill_assignment_policy = RoutedExpertAssignmentPolicy::StaticOwner;
    return domain;
}
/**
 * @brief Declare actual participant ownership through canonical topology/domain policies.
 * @param mode Exact CPU, GPU-overlay or explicitly composed pipeline request.
 * @return Immutable physical topology; source admission and PMA still prove materialization.
 * @throws std::invalid_argument For an unknown mode or invalid expert-domain plan.
 */
inline ModelParityTopologyDefinition qwen4ExpProductionTopology(Qwen4ExpParityTopology mode)
{
    if (mode == Qwen4ExpParityTopology::CPU)
        return {.test_id = "CPU0", .kind = ModelParityTopologyKind::SingleDevice,
                .participants = {{GlobalDeviceAddress::cpu(0), 0}}};
    if (mode == Qwen4ExpParityTopology::CPU2TP)
        return {.test_id = "CPU01_TP2", .kind = ModelParityTopologyKind::RankLocalTensorParallel,
                .participants = {{GlobalDeviceAddress::cpu(0), 0},
                                 {GlobalDeviceAddress::cpu(1), 0}},
                .collective = Collective::HOST};
    if (mode == Qwen4ExpParityTopology::CPU2PP)
        return {.test_id = "CPU0_CPU1_PP2", .kind = ModelParityTopologyKind::RankLocalPipelineParallel,
                .participants = {{GlobalDeviceAddress::cpu(0), 0},
                                 {GlobalDeviceAddress::cpu(1), 0}},
                .collective = Collective::HOST, .pipeline_stage_sizes = {1, 1},
                .pipeline_weights = {1.0f, 1.0f},
                .pipeline_tensor_parallel_collectives = {Collective::HOST, Collective::HOST}};
    if (mode == Qwen4ExpParityTopology::ROCm4TP2PP2)
        return {.test_id = "ROCm0123_TP2_PP2", .kind = ModelParityTopologyKind::RankLocalPipelineParallel,
                .participants = {{GlobalDeviceAddress::rocm(0), 0},
                                 {GlobalDeviceAddress::rocm(1), 0},
                                 {GlobalDeviceAddress::rocm(2), 0},
                                 {GlobalDeviceAddress::rocm(3), 0}},
                .collective = Collective::RCCL, .pipeline_stage_sizes = {2, 2},
                .pipeline_weights = {1.0f, 1.0f},
                .pipeline_tensor_parallel_collectives = {Collective::RCCL, Collective::RCCL}};
    if (mode == Qwen4ExpParityTopology::CUDA2CPU2TPPP ||
        mode == Qwen4ExpParityTopology::ROCm2CPU2TPPP)
    {
        const bool cuda = mode == Qwen4ExpParityTopology::CUDA2CPU2TPPP;
        // Sixteen GPU layers are an authored partition shared by both vendors,
        // not a capacity-driven substitution. Native PMA still admits the BOM.
        return {.test_id = cuda ? "CUDA01_CPU01_TP2_PP2" : "ROCm01_CPU01_TP2_PP2",
                .kind = ModelParityTopologyKind::RankLocalPipelineParallel,
                .participants = {{cuda ? GlobalDeviceAddress::cuda(0) : GlobalDeviceAddress::rocm(0), 0},
                                 {cuda ? GlobalDeviceAddress::cuda(1) : GlobalDeviceAddress::rocm(1), 0},
                                 {GlobalDeviceAddress::cpu(0), 0},
                                 {GlobalDeviceAddress::cpu(1), 0}},
                .collective = Collective::HETEROGENEOUS, .pipeline_stage_sizes = {2, 2},
                .pipeline_weights = {1.0f, 2.0f},
                .pipeline_tensor_parallel_collectives = {
                    cuda ? Collective::NCCL : Collective::RCCL, Collective::HOST}};
    }
    if (mode == Qwen4ExpParityTopology::CUDA2ROCm2CPU2TPPP)
        return {.test_id = "CUDA01_ROCm01_CPU01_TP2_PP3",
                .kind = ModelParityTopologyKind::RankLocalPipelineParallel,
                .participants = {{GlobalDeviceAddress::cuda(0), 0},
                                 {GlobalDeviceAddress::cuda(1), 0},
                                 {GlobalDeviceAddress::rocm(0), 0},
                                 {GlobalDeviceAddress::rocm(1), 0},
                                 {GlobalDeviceAddress::cpu(0), 0},
                                 {GlobalDeviceAddress::cpu(1), 0}},
                .collective = Collective::HETEROGENEOUS, .pipeline_stage_sizes = {2, 2, 2},
                .pipeline_weights = {2.0f, 3.0f, 7.0f},
                .pipeline_tensor_parallel_collectives = {
                    Collective::NCCL, Collective::RCCL, Collective::HOST}};
    if (mode == Qwen4ExpParityTopology::CUDAROCmCPU)
    {
        // The mixed-vendor boundary belongs to the public pipeline executor.
        // These diagnostic weights assign 8/12/28 complete layers. Sixteen
        // expert layers plus native fixed storage exceed a 32-GiB MI50; the
        // explicit split retains space for its captured serving family.
        // Admission still owns every byte and rejects undersized devices.
        // An extra GPU expert domain inside one continuation graph would violate
        // graph-local execution and cannot represent this pipeline.
        return {.test_id = "CUDA0_ROCm0_CPU0_PP",
                .kind = ModelParityTopologyKind::RankLocalPipelineParallel,
                .participants = {{GlobalDeviceAddress::cuda(0),0},
                                 {GlobalDeviceAddress::rocm(0),0},
                                 {GlobalDeviceAddress::cpu(0),0}},
                .collective = Collective::HETEROGENEOUS,
                .mpi_ranks = 1,
                .pipeline_stage_sizes = {1,1,1},
                .pipeline_weights = {2.0f,3.0f,7.0f}};
    }
    const bool cuda = mode == Qwen4ExpParityTopology::CUDACPU;
    if (!cuda && mode != Qwen4ExpParityTopology::ROCmCPU)
        throw std::invalid_argument("unknown Flash Next parity topology");
    auto plan = std::make_shared<MoERoutedExpertPlacementPlan>();
    plan->enabled = true;
    plan->topology = RoutedExpertPlacementTopology::TieredOverlay;
    plan->residency_policy = RoutedExpertResidencyPolicy::StaticById;
    plan->owner_order = RoutedExpertOwnerOrder::Ordinal;
    plan->domains.push_back(qwen4ExpParityDomain(cuda ? "cuda_continuation" : "rocm_continuation",
        cuda ? GlobalDeviceAddress::cuda(0) : GlobalDeviceAddress::rocm(0),
        cuda ? CollectiveBackendType::NCCL : CollectiveBackendType::RCCL));
    plan->domains.push_back(qwen4ExpParityDomain("cpu_capacity", GlobalDeviceAddress::cpu(0),
                                               CollectiveBackendType::HOST));
    plan->continuation_domain = plan->domains.front().name;
    plan->base_model_domain = plan->continuation_domain;
    plan->shared_expert_domain = plan->continuation_domain;
    plan->continuation_domain_spec.domain = plan->continuation_domain;
    plan->continuation_domain_spec.logical_root_participant = 0;
    plan->continuation_domain_spec.setDensePolicy(DenseParallelPolicy::Replicated);
    for (size_t index = 0; index < plan->domains.size(); ++index)
        plan->routed_tiers.push_back({.name = "tier_" + std::to_string(index),
            .domain = plan->domains[index].name, .priority = static_cast<int>(index) * 10,
            .max_experts_per_layer = 0, .memory_budget_bytes = 0,
            .fallback = index + 1 == plan->domains.size()});
    const auto validation = validateMoERoutedExpertPlacementPlan(*plan);
    if (!validation.ok()) throw std::invalid_argument("Flash Next parity expert topology is invalid");
    ModelParityTopologyDefinition topology{
        .test_id = cuda ? "CUDA0_CPU0_ExpertOverlay" : "ROCm0_CPU0_ExpertOverlay",
        .kind = ModelParityTopologyKind::RankLocalMultiDomain,
        .mpi_ranks = 1, .expert_overlay_plan = std::move(plan)};
    for (const auto &domain : topology.expert_overlay_plan->domains)
        topology.participants.push_back({domain.participants.front(), 0});
    return topology;
}
/** @brief Declare all actual fused checkpoints and strict complete-layer numerical gates. */
inline ModelParityDefinition qwen4ExpProductionDefinition(Qwen4ExpParityTopology topology)
{
    auto definition = qwen4ExpMainParityDefinition(qwen4ExpProductionTopology(topology),
        "parity-results/qwen-flash-next/reference/main-ram-mmap");
    // Pinned checkpoint tokenizer, revision de4b8e4d43b917e7706784d8bb445c9af86a3540.
    definition.model.token_ids = {760,3841,13477,37550,33075,888,279,15217,5388};
    definition.thresholds.early_layers_count = 48;
    definition.thresholds.min_early_layers_passed = 48;
    // Explicit provisional ordinary-logit regression limits; MTP keeps its strict inherited budget.
    definition.thresholds.final_logit_mean_kl = FinalLogitMeanKLContract{};
    definition.graph_snapshot_policy.enabled = true;
    definition.graph_snapshot_policy.require_graph_execution_on_gpu = true;
    definition.graph_snapshot_policy.require_snapshot_publication = true;
    auto &required = definition.graph_snapshot_policy.required_prefill_snapshot_keys;
    required = {"EMBEDDING", "LM_HEAD"};
    for (int layer = 0; layer < 48; ++layer)
    {
        const auto prefix = "layer" + std::to_string(layer) + "_";
        for (const auto *stage : {"GR_NORM", "GR_READ", "GR_WRITE", "FFN_NORM", "FFN_RESIDUAL"})
            required.push_back(prefix + stage);
        if ((layer + 1) % 4 == 0)
            for (const auto *stage : {"QSA_INDEXER_RAW_QUERY", "QSA_INDEXER_QUERY", "QSA_INDEXER_POOLED_KEYS", "QSA_SELECTED_INDICES"})
                required.push_back(prefix + stage);
    }
    for (const auto *stage : {"PLE_HASH_INDICES", "PLE_GATHER", "PLE_GATE", "PLE_CONVOLUTION"})
        required.push_back(std::string("layer1_") + stage);
    definition.graph_snapshot_policy.required_decode_snapshot_keys = required;
    definition.collective_evidence_source = ParityCollectiveEvidenceSource::PostCollectiveSnapshot;
    // A two-step diagnostic must observe a genuine economic publication;
    // an environment-owned clock or a long default window cannot prove it.
    MoERebalanceRuntimeConfig movement;
    movement.mode = MoERebalanceRuntimeMode::Dynamic;
    movement.window_size = movement.max_window_size = 1;
    movement.window_growth_factor = 1.0f;
    movement.dynamic_imbalance_threshold_per_mille = 1000;
    movement.dynamic_min_improvement_per_mille = 0;
    movement.dynamic_max_swaps_per_layer = movement.dynamic_max_plan_entries_per_wave = 20;
    movement.dynamic_min_window_activations = 0;
    movement.device_min_load_spread_improvement = 0;
    movement.device_min_load_spread_improvement_divisor = 0;
    movement.device_min_wave_spread_improvement_per_payload_slot = 0;
    movement.device_min_foreign_rows_per_critical_path_payload_slot = 0;
    movement.device_min_router_spread_improvement_per_payload_slot = 0;
    movement.device_max_post_wave_load_spread_per_mille = 1000;
    movement.device_maintenance_slack_tokens = 0;
    movement.device_min_maintenance_period_tokens = movement.device_initial_maintenance_period_tokens = 1;
    movement.migration_payoff_horizon_tokens = 65'536;
    movement.release_raw_expert_weights = false;
    definition.dynamic_rebalance = {.economic_movement = movement,
                                    .economic_movement_and_observed_speedup = movement};
    return definition;
}
/** @return Canonical four-topology product, including explicit RAM/mmap and standard expert axes. */
inline const std::vector<ModelParityCase> &qwen4ExpProductionCases()
{
    static const auto cases = [] {
        std::vector<ModelParityCase> result;
        for (const auto topology : {Qwen4ExpParityTopology::CPU, Qwen4ExpParityTopology::CUDACPU,
                                   Qwen4ExpParityTopology::ROCmCPU, Qwen4ExpParityTopology::CUDAROCmCPU})
        {
            auto expanded = expandModelParityDefinition(qwen4ExpProductionDefinition(topology));
            result.insert(result.end(), std::make_move_iterator(expanded.begin()), std::make_move_iterator(expanded.end()));
        }
        return result;
    }();
    return cases;
}
/**
 * @return Explicit adaptive host-storage cases using the SAME canonical numerical/source/topology definition.
 *
 * Keep the original20 whole-tier cases untouched. These20 additional model
 * gates inherit identical gold, snapshots, main48, prefix and movement evidence;
 * they acquire no numerical or serving certificate merely by being declared.
 */
inline const std::vector<ModelParityCase> &qwen4ExpHostAdaptiveProductionCases()
{
    static const auto cases=[] {
        std::vector<ModelParityCase> result;
        for(const auto topology:{Qwen4ExpParityTopology::CPU,Qwen4ExpParityTopology::CUDACPU,
            Qwen4ExpParityTopology::ROCmCPU,Qwen4ExpParityTopology::CUDAROCmCPU})
        {
            auto definition=qwen4ExpProductionDefinition(topology);
            definition.features.ngram_storage=qwen4ExpHostAdaptiveStorageAxes();
            auto expanded=expandModelParityDefinition(definition);
            result.insert(result.end(),std::make_move_iterator(expanded.begin()),std::make_move_iterator(expanded.end()));
        }
        return result;
    }();
    return cases;
}
} // namespace llaminar2::test::parity::qwen4exp
