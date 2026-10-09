/**
 * @file RankOrchestratorConfig.cpp
 * @brief Pure rank/PP configuration validation and stage-runtime projection.
 *
 * The production constructor and device-free regressions consume these same
 * methods. Stage projection preserves the authored pipeline and selects one
 * expert authority before any weight, collective or graph is materialized.
 * Keeping setup policy separate from native runner execution permits focused
 * ownership tests without loading a model or initializing a device backend.
 */
#include "RankOrchestrator.h"
#include "execution/mpi_orchestration/RankExecutionPlan.h"
#include "execution/moe/MoEOverlayPipelineStageBinding.h"
#include "utils/Logger.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace llaminar2
{
    bool RankOrchestrator::PPStageConfig::validate() const
    {
        // Layer range must be valid
        if (last_layer <= first_layer)
        {
            LOG_ERROR("PPStageConfig: Invalid layer range [" << first_layer << ", " << last_layer << ")");
            return false;
        }

        // Must have at least one device
        if (stage_devices.empty())
        {
            LOG_ERROR("PPStageConfig: No stage devices specified");
            return false;
        }

        // If TP weights are provided, must match device count
        if (!tp_weights.empty() && tp_weights.size() != stage_devices.size())
        {
            LOG_ERROR("PPStageConfig: TP weights count (" << tp_weights.size()
                                                          << ") doesn't match device count (" << stage_devices.size() << ")");
            return false;
        }

        // If TP weights are provided, must sum to approximately 1.0
        if (!tp_weights.empty())
        {
            float sum = std::accumulate(tp_weights.begin(), tp_weights.end(), 0.0f);
            if (std::abs(sum - 1.0f) > 0.01f)
            {
                LOG_ERROR("PPStageConfig: TP weights sum to " << sum << ", expected 1.0");
                return false;
            }
        }

        return true;
    }

    RankOrchestrator::ParallelismMode
    RankOrchestrator::Config::detectMode() const
    {
        if (pp_stages.empty())
        {
            // No PP stages - pure TP mode
            return ParallelismMode::TP;
        }

        // Check if any PP stage is a TP domain
        bool has_tp_stages = std::any_of(pp_stages.begin(), pp_stages.end(),
                                         [](const PPStageConfig &stage)
                                         { return stage.isTPDomain(); });

        return has_tp_stages ? ParallelismMode::TP_PP : ParallelismMode::PP;
    }

    std::vector<int> RankOrchestrator::Config::buildLayerBoundaries() const
    {
        std::vector<int> boundaries;
        if (pp_stages.empty())
        {
            return boundaries;
        }

        boundaries.push_back(0);
        for (const auto &stage : pp_stages)
        {
            boundaries.push_back(stage.last_layer);
        }
        return boundaries;
    }

    bool RankOrchestrator::Config::validate() const
    {
        ParallelismMode effective = effectiveMode();

        if (prepared_weight_admission ==
                PreparedWeightAdmission::ReuseCertifiedCompleteSet &&
            !prepared_weight_store)
        {
            LOG_ERROR(
                "RankOrchestrator::Config: certified prepared-weight reuse "
                "requires the exact model-owned PreparedWeightStore");
            return false;
        }

        if (effective == ParallelismMode::TP)
        {
            // TP mode validation
            if (devices.empty())
            {
                LOG_ERROR("RankOrchestrator::Config: No devices specified for TP mode");
                return false;
            }

            // If weights are provided, must match device count
            if (!weights.empty() && weights.size() != devices.size())
            {
                LOG_ERROR("RankOrchestrator::Config: Weights count (" << weights.size()
                                                                      << ") doesn't match device count (" << devices.size() << ")");
                return false;
            }

            // If weights are provided, must sum to approximately 1.0
            if (!weights.empty())
            {
                float sum = std::accumulate(weights.begin(), weights.end(), 0.0f);
                if (std::abs(sum - 1.0f) > 0.01f)
                {
                    LOG_ERROR("RankOrchestrator::Config: Weights sum to " << sum << ", expected 1.0");
                    return false;
                }
            }
        }
        else
        {
            // PP or TP_PP mode validation
            if (pp_stages.empty())
            {
                LOG_ERROR("RankOrchestrator::Config: No PP stages specified for PP mode");
                return false;
            }

            // Validate each stage
            for (size_t i = 0; i < pp_stages.size(); ++i)
            {
                if (!pp_stages[i].validate())
                {
                    LOG_ERROR("RankOrchestrator::Config: PP stage " << i << " validation failed");
                    return false;
                }
            }

            try
            {
                for (size_t i = 0; i < pp_stages.size(); ++i)
                    (void)forPipelineStage(i);
            }
            catch (const std::exception &error)
            {
                LOG_ERROR("RankOrchestrator::Config: invalid pipeline ownership: " << error.what());
                return false;
            }

            // Check layer continuity (no gaps)
            int expected_first = 0;
            for (size_t i = 0; i < pp_stages.size(); ++i)
            {
                if (pp_stages[i].first_layer != expected_first)
                {
                    LOG_ERROR("RankOrchestrator::Config: PP stage " << i
                                                                    << " first_layer=" << pp_stages[i].first_layer
                                                                    << " but expected " << expected_first << " (gap in layers)");
                    return false;
                }
                expected_first = pp_stages[i].last_layer;
            }

            // First stage should have embedding, last should have LM head
            if (!pp_stages.front().has_embedding)
            {
                LOG_WARN("RankOrchestrator::Config: First PP stage doesn't have embedding flag set");
            }
            if (!pp_stages.back().has_lm_head)
            {
                LOG_WARN("RankOrchestrator::Config: Last PP stage doesn't have lm_head flag set");
            }
        }

        return true;
    }

    std::vector<float> RankOrchestrator::Config::getNormalizedWeights() const
    {
        if (weights.empty() || weights.size() != devices.size())
        {
            // Equal distribution
            float equal_weight = 1.0f / static_cast<float>(devices.size());
            return std::vector<float>(devices.size(), equal_weight);
        }

        // Normalize to ensure sum is exactly 1.0
        float sum = std::accumulate(weights.begin(), weights.end(), 0.0f);
        if (sum <= 0.0f)
        {
            float equal_weight = 1.0f / static_cast<float>(devices.size());
            return std::vector<float>(devices.size(), equal_weight);
        }

        std::vector<float> normalized(weights.size());
        for (size_t i = 0; i < weights.size(); ++i)
        {
            normalized[i] = weights[i] / sum;
        }
        return normalized;
    }

    // =========================================================================
    // Config::fromPlan — Canonical translation from RankExecutionPlan
    // =========================================================================

    RankOrchestrator::Config
    RankOrchestrator::Config::fromRuntime(const RuntimeConfig &runtime)
    {
        Config config;

        // Runtime fields from pre-parsed RuntimeConfig
        config.max_seq_len = runtime.max_seq_len;
        config.resident_graph_rows = runtime.resident_graph_rows;
        config.batch_size = runtime.batch_size;
        config.activation_precision = runtime.activation_precision;
        config.kv_cache_precision = runtime.kv_cache_precision;
        config.tp_allreduce_precision_override =
            runtime.tp_allreduce_precision_override;
        config.prefix_cache = runtime.prefix_cache;
        config.mtp = runtime.mtp;
        config.routed_expert_compute_policy = runtime.routed_expert_compute_policy;
        config.routed_expert_owner_order =
            runtime.routed_expert_owner_order;
        config.moe_hot_expert_cache = runtime.moe_hot_expert_cache;
        config.moe_routed_prefill = runtime.moe_routed_prefill;
        config.moe_rebalance = runtime.moe_rebalance;

        config.fused_attention_backend = runtime.fused_attention_backend;
        config.kv_cache_scale_k = runtime.kv_cache_scale_k;
        config.kv_cache_scale_v = runtime.kv_cache_scale_v;
        return config;
    }

    RankOrchestrator::Config
    RankOrchestrator::Config::forPipelineStage(std::size_t stage_index) const
    {
        const auto &stage = pp_stages.at(stage_index);
        if (!stage.validate() || effectiveMode() == ParallelismMode::TP || nested_pp_stage_config)
            throw std::invalid_argument("Pipeline child requires one valid stage in a PP parent");
        if (moe_routed_expert_plan || moe_expert_overlay_residency_authority ||
            moe_expert_overlay_participant_residency || moe_expert_overlay_decode_histogram ||
            moe_expert_overlay_mpi_ctx || moe_rank_batch_transport_registry ||
            moe_device_controller_fabric || moe_node_local_route_exchange ||
            !moe_node_local_route_transport_policy.unresolved())
            throw std::invalid_argument("Pipeline parent cannot broadcast model-wide expert runtime; install each stage's binding");
        const bool has_expert_stage = std::any_of(pp_stages.begin(), pp_stages.end(),
            [](const auto &candidate) { return candidate.moe_runtime != nullptr; });
        if (has_expert_stage && std::any_of(pp_stages.begin(), pp_stages.end(),
                [](const auto &candidate) { return candidate.moe_runtime == nullptr; }))
            throw std::invalid_argument("Pipeline expert runtime installation must cover every stage before graph construction");

        Config child = *this;
        child.pp_stages.clear();
        child.mode = ParallelismMode::TP;
        child.devices = stage.stage_devices;
        child.weights = stage.tp_weights;
        child.backend = stage.tp_backend;
        child.nested_pp_stage_config = FactoryPPStageConfig{
            .first_layer = stage.first_layer, .last_layer = stage.last_layer,
            .has_embedding = stage.has_embedding, .has_lm_head = stage.has_lm_head};
        if (stage.moe_runtime)
        {
            stage.moe_runtime->requireDestination(*child.nested_pp_stage_config,
                child.devices, child.weights, child.backend, moe_rebalance.mode);
            // The authority's immutable initial snapshot supplies the plan.
            // Legacy graph configuration takes a mutable shared_ptr, so lend
            // a private metadata copy rather than exposing the live snapshot.
            child.moe_routed_expert_plan = std::make_shared<MoERoutedExpertPlacementPlan>(
                *stage.moe_runtime->plan());
            const auto &owners = stage.moe_runtime->owners();
            child.moe_expert_overlay_residency_authority = owners.authority;
            child.moe_expert_overlay_participant_residency = owners.residency;
            child.moe_expert_overlay_decode_histogram = owners.histogram;
            child.moe_expert_overlay_mpi_ctx = owners.mpi;
            child.moe_rank_batch_transport_registry = owners.rank_transport;
            child.moe_device_controller_fabric = owners.device_fabric;
            const auto &domains = child.moe_routed_expert_plan->domains;
            const auto continuation = std::find_if(domains.begin(), domains.end(),
                [&](const auto &domain) {
                    return domain.name == child.moe_routed_expert_plan->effectiveBaseModelDomain();
                });
            child.routed_expert_compute_policy = continuation->routed_compute_policy;
        }
        return child;
    }

    RankOrchestrator::Config
    RankOrchestrator::Config::fromPlan(const RankExecutionPlan &plan)
    {
        Config config = fromRuntime(plan.runtime);

        if (plan.usesLocalPP())
        {
            // PP mode: build stage configs from plan boundaries
            config.mode = ParallelismMode::PP;

            const auto &pp_devices = plan.local_pp_devices;
            const auto &boundaries = plan.local_pp_layer_boundaries;
            const auto &stage_tp_info = plan.local_pp_stage_tp_info;

            for (size_t i = 0; i < pp_devices.size(); ++i)
            {
                PPStageConfig stage_cfg;
                stage_cfg.first_layer = boundaries[i];
                stage_cfg.last_layer = boundaries[i + 1]; // exclusive
                stage_cfg.has_embedding = (i == 0);
                stage_cfg.has_lm_head = (i == pp_devices.size() - 1);

                // Use per-stage TP info if available (TP-in-PP composition)
                if (i < stage_tp_info.size() && stage_tp_info[i].devices.size() > 1)
                {
                    stage_cfg.stage_devices = stage_tp_info[i].devices;
                    stage_cfg.tp_weights = stage_tp_info[i].tp_weights;
                    stage_cfg.tp_backend = stage_tp_info[i].tp_backend;
                }
                else
                {
                    stage_cfg.stage_devices = {pp_devices[i]};
                }

                // Cross-vendor detection for host-staged hidden state transfer
                // Compare primary devices of adjacent stages
                auto primaryDeviceType = [&](size_t idx) -> DeviceType
                {
                    if (idx < stage_tp_info.size() && !stage_tp_info[idx].devices.empty())
                        return stage_tp_info[idx].devices[0].device_type;
                    if (idx < pp_devices.size())
                        return pp_devices[idx].device_type;
                    return DeviceType::CPU;
                };
                if (i + 1 < pp_devices.size() &&
                    primaryDeviceType(i) != primaryDeviceType(i + 1))
                {
                    // Cross-vendor PP stage boundary detected
                }

                config.pp_stages.push_back(std::move(stage_cfg));
            }
        }
        else
        {
            // TP mode: copy devices, weights, backend
            config.devices = plan.local_tp_devices;
            if (!plan.local_tp_weights.empty())
            {
                config.weights = plan.local_tp_weights;
            }
            config.backend = plan.local_tp_backend;
        }

        return config;
    }

}
