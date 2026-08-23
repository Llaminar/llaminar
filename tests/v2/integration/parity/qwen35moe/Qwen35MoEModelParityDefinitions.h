/**
 * @file Qwen35MoEModelParityDefinitions.h
 * @brief Authenticated Qwen3.5 MoE identities and ExpertOverlay topologies.
 *
 * Every Qwen3.5 MoE production parity binary consumes these declarations so
 * model identity, Hugging Face reference identity, numerical thresholds, and
 * routed-expert topology cannot drift between fixtures.  Tier direction is
 * expressed only by integer priority; names are diagnostic labels and carry
 * no placement semantics.
 */

#pragma once

#include "../ModelParityDefinition.h"

#include "execution/moe/MoERoutedExpertPlacementPlan.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace llaminar2::test::parity::qwen35moe
{
    /** Exact prompt shared by the authenticated 35B reference packs. */
    inline constexpr const char *kQwen35MoEParityPrompt =
        "The quick brown fox jumps over the lazy dog";

    /** @return Authenticated tokenization of @ref kQwen35MoEParityPrompt. */
    inline std::vector<int> qwen35MoEParityTokenIds()
    {
        return {760, 3841, 13477, 37550, 33075, 888, 279, 15217, 5388};
    }

    /** @return Canonical Qwen3.5-35B-A3B Q4_K_XL identity. */
    inline ModelParityModelDefinition qwen35MoE35BQ4KXLParityModel()
    {
        return ModelParityModelDefinition{
            .test_id = "Qwen35MoE_35B_Q4KXL",
            .model_path =
                "models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf",
            .reference_directory = "pytorch_qwen35_moe_snapshots",
            .prompt = kQwen35MoEParityPrompt,
            .token_ids = qwen35MoEParityTokenIds(),
            .decode_steps = 5,
            .max_seq_len = 4096,
            .transformer_layers = 40,
            .attention_heads = 16,
            .kv_heads = 2,
        };
    }

    /** @return Canonical smaller Q3_K_S GPU proof identity. */
    inline ModelParityModelDefinition qwen35MoE35BQ3KSParityModel()
    {
        auto model = qwen35MoE35BQ4KXLParityModel();
        model.test_id = "Qwen35MoE_35B_Q3KS";
        model.model_path = "models/Qwen3.5-35B-A3B-Q3_K_S.gguf";
        model.reference_directory = "pytorch_qwen35_moe_q3ks_snapshots";
        return model;
    }

    /** @return Strict single-device numerical contract shared by all backends. */
    inline BackendThresholds qwen35MoESingleDeviceThresholds()
    {
        return {
            .cosine_threshold = 0.96f,
            .decode_cosine_threshold = 0.98f,
            .early_layers_count = 6,
            .min_early_layers_passed = 5,
            .kl_threshold = 0.03f,
            .min_top1_accuracy = 80.0f,
            .min_top5_accuracy = 60.0f,
            .pytorch_top1_in_topk = 3,
        };
    }

    /** @return Shard-aware numerical contract for routed multi-device graphs. */
    inline BackendThresholds qwen35MoEMultiDeviceThresholds()
    {
        return {
            .cosine_threshold = 0.90f,
            .decode_cosine_threshold = 0.80f,
            .early_layers_count = 6,
            .min_early_layers_passed = 5,
            .kl_threshold = 0.05f,
            .excluded_stages = {
                "Q_PROJECTION",
                "K_PROJECTION",
                "V_PROJECTION",
                "Q_NORM",
                "K_NORM",
                "Q_ROPE",
                "K_ROPE",
                "ATTENTION_CONTEXT",
                "FA_GATE",
                "ATTENTION_CONTEXT_GATED",
                "FFN_GATE",
                "FFN_UP",
                "FFN_SWIGLU",
                "QKV_PROJECTION",
                "GDN_CONV1D_OUTPUT",
                "GDN_Z_PROJECTION",
                "GDN_DELTA_RULE_OUTPUT",
                "GDN_NORM_GATE_OUTPUT",
                /* Branch-local partials are not canonical post-collective values. */
                "MOE_EXPERT_OUTPUT",
                "MOE_SHARED_EXPERT_OUTPUT",
                "MOE_SHARED_GATE_OUTPUT",
            },
            .allreduce_stages = {},
            .min_top1_accuracy = 80.0f,
            .min_top5_accuracy = 80.0f,
            .pytorch_top1_in_topk = 4,
        };
    }

    /**
     * @brief Build one homogeneous single-domain ExpertOverlay authority.
     *
     * Dense/shared work remains tensor parallel while routed experts are
     * apportioned across the same participants. A zero quota and budget ask
     * production capacity planning to fill the only fallback tier up to its
     * computed safety margin rather than embedding a model-specific cap.
     *
     * @param domain_name Stable diagnostic domain identity.
     * @param scope Rank-local or node-local execution scope.
     * @param backend Exact collective for the homogeneous participants.
     * @param participants Physical devices in logical participant order.
     * @return Valid immutable blueprint for typed matrix expansion.
     */
    inline std::shared_ptr<const MoERoutedExpertPlacementPlan>
    qwen35MoESingleDomainOverlayPlan(
        std::string domain_name,
        ExecutionDomainScope scope,
        CollectiveBackendType backend,
        std::vector<GlobalDeviceAddress> participants)
    {
        RoutedExpertDomain domain;
        domain.name = domain_name;
        domain.scope = scope;
        domain.backend = backend;
        domain.participants = std::move(participants);
        domain.routed_compute_policy = RoutedExpertComputePolicy::Apportioned;
        domain.routed_phase_policy = RoutedExpertPhasePolicy::Uniform;
        domain.routed_decode_assignment_policy =
            RoutedExpertAssignmentPolicy::StaticOwner;
        domain.routed_prefill_assignment_policy =
            RoutedExpertAssignmentPolicy::StaticOwner;

        auto plan = std::make_shared<MoERoutedExpertPlacementPlan>();
        plan->enabled = true;
        plan->topology = RoutedExpertPlacementTopology::SingleDomain;
        plan->continuation_domain = domain_name;
        plan->base_model_domain = domain_name;
        plan->shared_expert_domain = domain_name;
        plan->continuation_domain_spec.domain = domain_name;
        plan->continuation_domain_spec.logical_root_participant = 0;
        plan->continuation_domain_spec.setDensePolicy(
            DenseParallelPolicy::TensorParallel);
        plan->continuation_domain_spec.hidden_layout =
            MoEContinuationActivationLayout::ReplicatedHidden;
        plan->residency_policy = RoutedExpertResidencyPolicy::StaticById;
        plan->owner_order = RoutedExpertOwnerOrder::Ordinal;
        plan->dense_domains = {domain.toExecutionDomainDefinition()};
        plan->domains = {std::move(domain)};
        plan->routed_tiers = {{
            .name = "priority_0",
            .domain = domain_name,
            .priority = 0,
            .max_experts_per_layer = 0,
            .memory_budget_bytes = 0,
            .fallback = true,
        }};

        const auto validation = validateMoERoutedExpertPlacementPlan(*plan);
        if (!validation.ok())
        {
            std::string message =
                "invalid Qwen3.5 MoE single-domain parity plan";
            for (const auto &error : validation.errors)
                message += "\n - " + error;
            throw std::logic_error(message);
        }
        return plan;
    }

    /** @return Two-ROCm rank-local ExpertOverlay topology. */
    inline ModelParityTopologyDefinition qwen35MoERocm2LocalTPTopology()
    {
        return {
            .test_id = "LocalTP_RCCL_2xROCm_ExpertOverlay",
            .kind = ModelParityTopologyKind::RankLocalTensorParallel,
            .participants = {
                {GlobalDeviceAddress::rocm(0), 0},
                {GlobalDeviceAddress::rocm(1), 0},
            },
            .collective = Collective::RCCL,
            .mpi_ranks = 1,
            .expert_overlay_plan = qwen35MoESingleDomainOverlayPlan(
                "qwen35_moe_rocm_local_tp",
                ExecutionDomainScope::RANK_LOCAL,
                CollectiveBackendType::RCCL,
                {GlobalDeviceAddress::rocm(0), GlobalDeviceAddress::rocm(1)}),
        };
    }

    /** @return Two-rank/two-NUMA CPU ExpertOverlay topology. */
    inline ModelParityTopologyDefinition qwen35MoECPU2NodeTPTopology()
    {
        return {
            .test_id = "NodeTP_2xMPI_CPU_ExpertOverlay",
            .kind = ModelParityTopologyKind::NodeTensorParallel,
            .participants = {
                {GlobalDeviceAddress::cpu(0), 0},
                {GlobalDeviceAddress::cpu(1), 1},
            },
            .collective = Collective::MPI,
            .mpi_ranks = 2,
            .expert_overlay_plan = qwen35MoESingleDomainOverlayPlan(
                "qwen35_moe_cpu_node_tp",
                ExecutionDomainScope::NODE_LOCAL,
                CollectiveBackendType::MPI,
                {GlobalDeviceAddress::cpu(0), GlobalDeviceAddress::cpu(1)}),
        };
    }

    /** @return Short-test economics that still require measured positive payoff. */
    inline MoERebalanceRuntimeConfig qwen35MoEDynamicParityEconomics()
    {
        MoERebalanceRuntimeConfig config;
        config.mode = MoERebalanceRuntimeMode::Dynamic;
        config.window_size = 1;
        config.max_window_size = 1;
        config.window_growth_factor = 1.0f;
        config.dynamic_imbalance_threshold_per_mille = 1000;
        config.dynamic_min_improvement_per_mille = 0;
        config.dynamic_max_swaps_per_layer = 20;
        config.dynamic_max_plan_entries_per_wave = 20;
        config.dynamic_min_window_activations = 0;
        config.device_min_load_spread_improvement = 0;
        config.device_min_load_spread_improvement_divisor = 0;
        config.device_min_wave_spread_improvement_per_payload_slot = 0;
        config.device_min_foreign_rows_per_critical_path_payload_slot = 0;
        config.device_min_router_spread_improvement_per_payload_slot = 0;
        config.device_max_post_wave_load_spread_per_mille = 1000;
        config.device_maintenance_slack_tokens = 0;
        config.device_min_maintenance_period_tokens = 1;
        config.device_initial_maintenance_period_tokens = 1;
        config.migration_payoff_horizon_tokens = 65'536;
        config.release_raw_expert_weights = false;
        return config;
    }

    /**
     * @brief Join one model and topology to the standard precision/policy axes.
     */
    inline ModelParityDefinition qwen35MoEParityDefinition(
        ModelParityModelDefinition model,
        ModelParityTopologyDefinition topology,
        BackendThresholds thresholds)
    {
        ModelParityDefinition definition;
        definition.model = std::move(model);
        definition.topology = std::move(topology);
        definition.thresholds = std::move(thresholds);
        definition.precisions.activation = {ActivationPrecision::FP32};
        definition.precisions.kv_cache = {KVCachePrecision::FP16};
        definition.dynamic_rebalance = qwen35MoEDynamicParityEconomics();
        definition.collective_evidence_source =
            ParityCollectiveEvidenceSource::PostCollectiveSnapshot;
        return definition;
    }
} // namespace llaminar2::test::parity::qwen35moe
