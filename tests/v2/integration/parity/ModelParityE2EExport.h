/**
 * @file ModelParityE2EExport.h
 * @brief Read-only HTTP certification projection of canonical parity cells.
 *
 * GoogleTest parameter discovery publishes machine-readable metadata without
 * loading weights. The E2E driver consumes these records, never test-name
 * fragments or a second model/topology table. Arguments remain an argv vector
 * all the way to the existing HTTP harness; they are never shell-evaluated.
 */
#pragma once

#include <iomanip>
#include <ostream>
#include <sstream>

namespace llaminar2::test::parity
{
    /**
     * @brief Serialize the cell-owned movement obligation, not a runtime default.
     * @param evidence Existing typed numerical-cell movement requirement.
     * @return Stable cross-language certification spelling.
     * @throws std::invalid_argument if an invalid enum reaches discovery.
     *
     * A single-device cell may retain Dynamic policy defaults without having
     * another participant. Only the matrix authority can declare which proof
     * applies; HTTP validators must not reconstruct that fact from CLI defaults.
     */
    inline const char *modelParityE2EMovementEvidenceName(ModelParityMovementEvidence evidence)
    {
        switch (evidence)
        {
        case ModelParityMovementEvidence::NotApplicable: return "not_applicable";
        case ModelParityMovementEvidence::NoMovement: return "forbidden";
        case ModelParityMovementEvidence::MovementRequired: return "required";
        }
        throw std::invalid_argument("invalid model parity movement evidence");
    }

    /** @return JSON string literal with control characters escaped. */
    inline std::string modelParityJsonString(const std::string &value)
    {
        std::ostringstream out;
        out << '"';
        for (const unsigned char ch : value)
        {
            if (ch == '"' || ch == '\\') out << '\\' << ch;
            else if (ch < 32)
                out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                    << static_cast<unsigned int>(ch) << std::dec;
            else out << ch;
        }
        out << '"';
        return out.str();
    }

    /** @return Exact public CLI domain spelling, preserving ownership and policy. */
    inline std::string modelParityDomainArgument(const ExecutionDomainDefinition &domain)
    {
        std::ostringstream out;
        out << domain.name << '=';
        for (std::size_t i = 0; i < domain.participants.size(); ++i)
            out << (i ? "," : "") << domain.participants[i].toString();
        out << ";scope=" << executionDomainScopeToString(domain.scope)
            << ";backend=" << collectiveBackendTypeToString(domain.backend);
        if (domain.owner_rank) out << ";owner=" << *domain.owner_rank;
        if (!domain.ranks.empty())
        {
            out << ";ranks=";
            for (std::size_t i = 0; i < domain.ranks.size(); ++i)
                out << (i ? "," : "") << domain.ranks[i];
        }
        if (!domain.weights.empty())
        {
            out << ";weights=";
            for (std::size_t i = 0; i < domain.weights.size(); ++i)
                out << (i ? "," : "") << std::setprecision(9) << domain.weights[i];
        }
        if (domain.routed_compute_policy != RoutedExpertComputePolicy::Unspecified)
            out << ";routed_compute=" << routedExpertComputePolicyToString(domain.routed_compute_policy);
        if (domain.routed_phase_policy != RoutedExpertPhasePolicy::Unspecified)
            out << ";routed_phase=" << routedExpertPhasePolicyToString(domain.routed_phase_policy);
        if (domain.routed_decode_assignment_policy != RoutedExpertAssignmentPolicy::Unspecified)
            out << ";routed_decode_assignment=" << routedExpertAssignmentPolicyToString(domain.routed_decode_assignment_policy);
        if (domain.routed_prefill_assignment_policy != RoutedExpertAssignmentPolicy::Unspecified)
            out << ";routed_prefill_assignment=" << routedExpertAssignmentPolicyToString(domain.routed_prefill_assignment_policy);
        return out.str();
    }

    /**
     * @brief Project a tagged case onto the Release server's public CLI.
     * @param cell Existing expanded parity configuration, not a new matrix.
     * @return Argument vector excluding executable, subcommand, model and port.
     * @throws std::invalid_argument for untagged or non-exportable state.
     *
     * Context geometry belongs to the certification profile. Placement, MTP,
     * precision and economic policy come from the same production config as
     * numerical parity. Resolved runtime placements cannot be exported as
     * declarative startup intent.
     */
    inline std::vector<std::string> modelParityE2EServerArguments(const ModelParityCase &cell)
    {
        if (!cell.e2e_certification)
            throw std::invalid_argument("cannot export an untagged E2E cell");
        const auto config = cell.makeOrchestrationConfig(cell.model.model_path, 0);
        std::vector<std::string> args;
        const auto add = [&](const char *flag, const auto &value)
        {
            std::ostringstream out;
            out << std::setprecision(9) << value;
            args.insert(args.end(), {flag, out.str()});
        };
        add("--mpi-procs", cell.topology.mpi_ranks);
        add("--activation-precision", config.activation_precision);
        add("--kv-cache-precision", config.kv_cache_precision);
        add("--backend", collectiveBackendTypeToString(config.default_backend));
        if (!config.tp_allreduce_precision_override.empty())
            add("--tp-allreduce-precision", config.tp_allreduce_precision_override);
        args.push_back("--prefix-cache");
        add("--prefix-cache-storage", "tiered");
        add("--prefix-cache-terminal-state", "auto");

        if (cell.topology.isExpertOverlay())
        {
            const auto &plan = *config.moe_routed_expert_plan;
            if (!plan.placements.empty() || !plan.initial_layer_order_overrides.empty() ||
                plan.continuation_domain_spec.logical_root_participant != 0)
                throw std::invalid_argument("E2E export requires declarative CLI-representable overlay placement");
            add("--moe-routed-expert-placement", plan.topology == RoutedExpertPlacementTopology::SingleDomain
                ? "single-domain" : "tiered-overlay");
            add("--moe-routed-expert-continuation-domain", plan.continuation_domain);
            if (!plan.base_model_domain.empty()) add("--moe-routed-expert-base-model-domain", plan.base_model_domain);
            if (!plan.shared_expert_domain.empty()) add("--moe-routed-expert-shared-domain", plan.shared_expert_domain);
            add("--moe-continuation-dense-policy", denseParallelPolicyToString(plan.continuation_domain_spec.effectiveDensePolicy()));
            add("--moe-routed-expert-owner-order", cell.expert_overlay->owner_order == RoutedExpertOwnerOrder::Ordinal ? "ordinal" : "random");
            add("--moe-routed-expert-residency", cell.expert_overlay->movement == ModelParityExpertMovement::Static ? "static-by-id" : "rebalanced");
            for (const auto &domain : plan.dense_domains)
                add("--define-domain", modelParityDomainArgument(domain));
            for (const auto &domain : plan.domains)
                add("--moe-routed-expert-domain", modelParityDomainArgument(domain.toExecutionDomainDefinition()));
            for (const auto &tier : plan.routed_tiers)
            {
                if (tier.memory_budget_bytes % (1024 * 1024) != 0 || !tier.resolved_live_experts_per_layer.empty())
                    throw std::invalid_argument("E2E tier budget must be declarative and exactly expressible in MiB");
                add("--moe-routed-expert-tier", tier.name + "@" + tier.domain +
                    ";priority=" + std::to_string(tier.priority) +
                    ";max-experts-per-layer=" + std::to_string(tier.max_experts_per_layer) +
                    ";memory-mb=" + (tier.memory_budget_bytes ? std::to_string(tier.memory_budget_bytes / (1024 * 1024)) : "auto"));
            }
        }
        else
        {
            if (cell.topology.kind == ModelParityTopologyKind::SingleDevice)
                add("--device", config.device_for_this_rank.value().toString());
            if (!config.device_map.empty())
            {
                std::string map;
                for (const auto &[rank, address] : config.device_map)
                    map += (map.empty() ? "" : ",") + std::to_string(rank) + "=" + address.toString();
                add("--device-map", map);
            }
            if (!config.tp_devices.empty())
            {
                std::string devices;
                for (const auto &address : config.tp_devices)
                    devices += (devices.empty() ? "" : ",") + address.toString();
                add("--tp-devices", devices);
            }
            add("--tensor-parallelism-degree", config.tp_degree);
            add("--tp-scope", tpScopeToString(config.tp_scope));
            for (const auto &domain : config.domain_definitions)
                add("--define-domain", modelParityDomainArgument(domain.toExecutionDomainDefinition()));
            for (const auto &stage : config.pp_stage_definitions)
                add("--pp-stage", std::to_string(stage.stage_id) + "=" + stage.domain_name + ":" +
                    std::to_string(stage.first_layer) + "-" + std::to_string(stage.last_layer));
        }
        // Off cells still retain the model's setup envelope, just as numerical
        // parity does. Dropping it here could change automatic expert capacity.
        add("--mtp-graph-capacity-draft-tokens", config.mtp.graph_capacity_draft_tokens);
        if (cell.mtpEnabled())
        {
            args.push_back("--mtp");
            add("--mtp-draft-tokens", config.mtp.draft_tokens);
            add("--mtp-verify-mode", "greedy");
            if (cell.usesDynamicMTPDepth())
            {
                add("--mtp-depth-policy", "dynamic");
                add("--mtp-min-draft-tokens", config.mtp.depth_policy.min_depth);
                add("--mtp-max-draft-tokens", config.mtp.depth_policy.max_depth);
                add("--mtp-initial-draft-tokens", config.mtp.depth_policy.initial_depth);
                add("--mtp-depth-window", config.mtp.depth_policy.window_size);
                add("--mtp-depth-min-samples", config.mtp.depth_policy.min_samples);
                add("--mtp-depth-cooldown", config.mtp.depth_policy.cooldown_steps);
                add("--mtp-depth-promote-windows", config.mtp.depth_policy.promote_consecutive_windows);
            }
        }
        add("--moe-hot-expert-cache", config.moe_hot_expert_cache.toString());
        add("--moe-residency-maintenance", moeRebalanceRuntimeModeToString(config.moe_rebalance.mode));
        // Every CLI-exposed economic scalar remains attached to its case. In
        // particular the HTTP harness must not invent a forced-movement policy.
#define LLAMINAR_E2E_ECONOMY(flag, field) add(flag, config.moe_rebalance.field)
        LLAMINAR_E2E_ECONOMY("--moe-residency-maintenance-window", window_size);
        LLAMINAR_E2E_ECONOMY("--moe-residency-maintenance-max-window", max_window_size);
        LLAMINAR_E2E_ECONOMY("--moe-residency-maintenance-window-growth", window_growth_factor);
        LLAMINAR_E2E_ECONOMY("--moe-migration-payoff-horizon-tokens", migration_payoff_horizon_tokens);
        LLAMINAR_E2E_ECONOMY("--moe-migration-transfer-slots", migration_transfer_slots);
        if (config.moe_rebalance.migration_execution_streams)
            add("--moe-migration-execution-streams", *config.moe_rebalance.migration_execution_streams);
        if (config.moe_rebalance.migration_cycles_per_wave)
            add("--moe-migration-cycles-per-wave", *config.moe_rebalance.migration_cycles_per_wave);
        LLAMINAR_E2E_ECONOMY("--moe-dynamic-imbalance-threshold-permille", dynamic_imbalance_threshold_per_mille);
        LLAMINAR_E2E_ECONOMY("--moe-dynamic-min-improvement-permille", dynamic_min_improvement_per_mille);
        LLAMINAR_E2E_ECONOMY("--moe-dynamic-max-swaps-per-layer", dynamic_max_swaps_per_layer);
        LLAMINAR_E2E_ECONOMY("--moe-dynamic-max-plan-entries-per-wave", dynamic_max_plan_entries_per_wave);
        LLAMINAR_E2E_ECONOMY("--moe-dynamic-min-window-activations", dynamic_min_window_activations);
        // Negative values mean unresolved production defaults, not negative
        // token counts accepted by CLI admission.
        if (config.moe_rebalance.device_maintenance_slack_tokens >= 0)
            LLAMINAR_E2E_ECONOMY("--moe-device-rebalance-maintenance-slack-tokens", device_maintenance_slack_tokens);
        if (config.moe_rebalance.device_min_maintenance_period_tokens >= 0)
            LLAMINAR_E2E_ECONOMY("--moe-device-rebalance-min-maintenance-period-tokens", device_min_maintenance_period_tokens);
        if (config.moe_rebalance.device_initial_maintenance_period_tokens >= 0)
            LLAMINAR_E2E_ECONOMY("--moe-device-rebalance-initial-maintenance-period-tokens", device_initial_maintenance_period_tokens);
        LLAMINAR_E2E_ECONOMY("--moe-device-rebalance-min-load-spread-improvement", device_min_load_spread_improvement);
        LLAMINAR_E2E_ECONOMY("--moe-device-rebalance-min-load-spread-improvement-divisor", device_min_load_spread_improvement_divisor);
        LLAMINAR_E2E_ECONOMY("--moe-device-rebalance-min-wave-spread-improvement-per-payload-slot", device_min_wave_spread_improvement_per_payload_slot);
        LLAMINAR_E2E_ECONOMY("--moe-device-rebalance-min-foreign-rows-per-critical-path-payload-slot", device_min_foreign_rows_per_critical_path_payload_slot);
        LLAMINAR_E2E_ECONOMY("--moe-device-rebalance-min-router-spread-improvement-per-payload-slot", device_min_router_spread_improvement_per_payload_slot);
        LLAMINAR_E2E_ECONOMY("--moe-device-rebalance-max-post-wave-load-spread-permille", device_max_post_wave_load_spread_per_mille);
#undef LLAMINAR_E2E_ECONOMY
        if (config.moe_rebalance.release_raw_expert_weights)
            args.push_back("--moe-release-raw-expert-weights");
        add("--moe-routed-prefill-assignment-window", config.moe_routed_prefill.assignment_window_tokens);
        add("--moe-overlay-prefill-segment-rows", config.moe_routed_prefill.overlay_segment_rows);
        add("--moe-routed-prefill-least-loaded-min-routed-rows", config.moe_routed_prefill.least_loaded_min_routed_rows);
        add("--moe-routed-prefill-llep-alpha-numerator", config.moe_routed_prefill.llep_alpha_numerator);
        add("--moe-routed-prefill-llep-alpha-denominator", config.moe_routed_prefill.llep_alpha_denominator);
        add("--moe-routed-prefill-llep-lambda-numerator", config.moe_routed_prefill.llep_lambda_numerator);
        add("--moe-routed-prefill-llep-lambda-denominator", config.moe_routed_prefill.llep_lambda_denominator);
        if (!config.moe_routed_prefill.llep_enable_balanced_skip)
            args.push_back("--moe-routed-prefill-llep-disable-balanced-skip");
        if (cell.prefill_graph.isSegmentedCaptured())
            add("--moe-overlay-prefill-segment-rows", cell.prefill_graph.captured_rows);
        return args;
    }

    /**
     * @brief GoogleTest's ADL parameter printer is the discovery export boundary.
     * @param cell Typed parameter, also consumed by the numerical fixture.
     * @param out GoogleTest discovery stream; emits one complete JSON line.
     */
    inline void PrintTo(const ModelParityCase &cell, std::ostream *out)
    {
        *out << "{\"model_parity_schema\":1,\"id\":" << modelParityJsonString(cell.testName())
             << ",\"model\":" << modelParityJsonString(cell.model.model_path) << ",\"e2e\":";
        if (!cell.e2e_certification) { *out << "null}"; return; }
        const auto &profile = *cell.e2e_certification;
        *out << "{\"context_length\":" << profile.context_length
             << ",\"minimum_prompt_tokens\":" << profile.minimum_prompt_tokens
             << ",\"generation_tokens\":" << profile.generation_tokens
             << ",\"request_timeout_seconds\":" << profile.request_timeout_seconds
             << ",\"readiness_timeout_seconds\":" << profile.readiness_timeout_seconds
             << ",\"thinking_modes\":" << modelParityJsonString(modelParityE2EThinkingModesName(profile.thinking_modes))
             << ",\"movement_evidence\":" << modelParityJsonString(modelParityE2EMovementEvidenceName(cell.movementEvidence()))
             << ",\"server_args\":[";
        const auto args = modelParityE2EServerArguments(cell);
        for (std::size_t i = 0; i < args.size(); ++i)
            *out << (i ? "," : "") << modelParityJsonString(args[i]);
        *out << "]}}";
    }
} // namespace llaminar2::test::parity
