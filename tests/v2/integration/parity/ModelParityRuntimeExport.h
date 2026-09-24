/**
 * @file ModelParityRuntimeExport.h
 * @brief Shared production-runtime projection of every canonical model cell.
 *
 * GoogleTest parameter discovery publishes machine-readable metadata without
 * loading weights. Generation regression and tagged E2E consume these records,
 * never test-name fragments or a second model/topology table. Arguments remain an argv vector
 * all the way to the existing HTTP harness; they are never shell-evaluated.
 */
#pragma once

#include <iomanip>
#include <map>
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

    /**
     * @return Main-model restore obligation, unchanged by topology or MTP policy.
     * @throws std::invalid_argument for an omitted or invalid model declaration.
     */
    inline const char *modelParityPrefixStateName(ModelParityPrefixState state)
    {
        switch (state)
        {
        case ModelParityPrefixState::AttentionKV: return "attention_kv";
        case ModelParityPrefixState::HybridRecurrent: return "hybrid_recurrent";
        default: throw std::invalid_argument("missing or invalid model parity prefix-state contract");
        }
    }

    /** @return Exact MTP policy without requiring consumers to parse cell names. */
    inline const char *modelParityMTPPolicyName(ModelParityMTP policy)
    {
        switch (policy)
        {
        case ModelParityMTP::Off: return "off";
        case ModelParityMTP::Depth1: return "depth_1";
        case ModelParityMTP::Depth2: return "depth_2";
        case ModelParityMTP::Depth3: return "depth_3";
        case ModelParityMTP::Depth15: return "depth_15";
        case ModelParityMTP::DynamicDepth: return "dynamic";
        }
        throw std::invalid_argument("invalid generation MTP policy");
    }

    /**
     * @return HTTP verifier policy supporting both greedy and stochastic requests.
     *
     * Numerical fixtures own their reference sampling policy. Public server
     * workloads must admit their actual request sampler: the stochastic policy
     * includes the greedy specialization, whereas a greedy-only verifier cannot
     * execute the fixed-seed stochastic generation probes. No runner overrides
     * this projection or substitutes an MTP-off path.
     */
    inline constexpr MTPVerifyMode modelParityHttpMTPVerifyMode() noexcept
    {
        return MTPVerifyMode::SpeculativeSampling;
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

    /** @brief Placement projection, independent of the model's runtime policies. */
    enum class ModelParityRuntimePlacementProjection
    {
        DeclaredCell,
        AutomaticCell,
        AutomaticRemoteCPUOverlay,
    };

    /** @return The existing cell's strategy family, without fixed device/rank placement. */
    inline OrchestrationStrategy modelParityAutomaticStrategy(const ModelParityCase &cell)
    {
        if (cell.topology.isExpertOverlay())
            return cell.topology.expert_overlay_plan->domains.size() == 1
                ? OrchestrationStrategy::TensorParallel : OrchestrationStrategy::ExpertOverlay;
        switch (cell.topology.kind)
        {
        case ModelParityTopologyKind::SingleDevice: return OrchestrationStrategy::SingleDevice;
        case ModelParityTopologyKind::RankLocalTensorParallel:
        case ModelParityTopologyKind::NodeTensorParallel:
        case ModelParityTopologyKind::GlobalTensorParallel: return OrchestrationStrategy::TensorParallel;
        case ModelParityTopologyKind::RankLocalPipelineParallel:
        case ModelParityTopologyKind::NodePipelineParallel: return OrchestrationStrategy::PipelineParallel;
        default: throw std::invalid_argument("canonical E2E topology has no automatic strategy projection");
        }
    }

    /** @return Exact compute cardinality from the sole typed topology declaration. */
    inline std::map<std::string, int> modelParityAutomaticDeviceCounts(const ModelParityCase &cell)
    {
        std::map<std::string, int> counts;
        for (const auto &participant : cell.topology.participants)
        {
            const auto &address = participant.address;
            if (!address.isCPU() && !address.isCUDA() && !address.isROCm())
                throw std::invalid_argument("canonical E2E has an unsupported compute backend");
            ++counts[address.isCPU() ? "cpu" : address.isCUDA() ? "cuda" : "rocm"];
        }
        if (counts.empty()) throw std::invalid_argument("canonical E2E requires compute participants");
        return counts;
    }

    /**
     * @brief Project any canonical case onto the Release server's public CLI.
     * @param cell Existing expanded parity configuration, not a new matrix.
     * @param placement Keep declared placement or request the tagged remote overlay.
     * @return Argument vector excluding executable, subcommand, model and port.
     * @throws std::invalid_argument for non-exportable state.
     *
     * The caller supplies its typed workload's context geometry. Placement, MTP,
     * precision and economic policy come from the same production config as
     * numerical parity. Resolved runtime placements cannot be exported as
     * declarative startup intent.
     */
    inline std::vector<std::string> modelParityServerArguments(
        const ModelParityCase &cell,
        ModelParityRuntimePlacementProjection placement = ModelParityRuntimePlacementProjection::DeclaredCell)
    {
        auto config = cell.makeOrchestrationConfig(cell.model.model_path, 0);
        const bool automatic_remote = placement == ModelParityRuntimePlacementProjection::AutomaticRemoteCPUOverlay;
        const bool declared = placement == ModelParityRuntimePlacementProjection::DeclaredCell;
        if (!automatic_remote && !declared && placement != ModelParityRuntimePlacementProjection::AutomaticCell)
            throw std::invalid_argument("invalid model parity placement projection");
        if (automatic_remote)
        {
            if (cell.remote_cpu_overlays.empty() || !cell.e2e_certification ||
                cell.topology.kind != ModelParityTopologyKind::SingleDevice ||
                cell.topology.participants.size() != 1 ||
                (!cell.topology.participants.front().address.isCUDA() &&
                 !cell.topology.participants.front().address.isROCm()))
                throw std::invalid_argument("automatic remote projection requires a tagged single-GPU cell");
            config.moe_rebalance = cell.dynamic_rebalance;
            config.moe_rebalance.mode = MoERebalanceRuntimeMode::Dynamic;
        }
        std::vector<std::string> args;
        const auto add = [&](const char *flag, const auto &value)
        {
            std::ostringstream out;
            out << std::setprecision(9) << value;
            args.insert(args.end(), {flag, out.str()});
        };
        if (declared || cell.topology.mpi_ranks > 1)
            add("--mpi-procs", cell.topology.mpi_ranks);
        add("--activation-precision", config.activation_precision);
        add("--kv-cache-precision", config.kv_cache_precision);
        if (declared) add("--backend", collectiveBackendTypeToString(config.default_backend));
        if (!config.tp_allreduce_precision_override.empty())
            add("--tp-allreduce-precision", config.tp_allreduce_precision_override);
        args.push_back("--prefix-cache");
        add("--prefix-cache-storage", "tiered");
        add("--prefix-cache-terminal-state", "auto");

        if (automatic_remote)
        {
            // Both routes deliberately exercise default auto: no --auto flag,
            // device map, local CPU indices or borrowed domain declaration.
            // Provisioning binds the exact requested endpoints and hostfile;
            // backend filters alone do not prove that remote work took place.
            add("--only-backends", cell.topology.participants.front().address.isCUDA() ? "cuda,cpu" : "rocm,cpu");
            add("--only-strategies", "expert-overlay");
            add("--auto-hosts", "all");
        }
        else if (!declared)
        {
            // This is a search constraint, not a prebuilt apply document. Auto
            // retains endpoint, rank, layer-split, domain and tier-role choices.
            // Requiring counts prevents a two-device E2E cell silently passing
            // as a faster single-device candidate on a larger machine.
            args.push_back("--auto");
            std::string backends, cardinality;
            for (const auto &[backend, count] : modelParityAutomaticDeviceCounts(cell))
            {
                backends += (backends.empty() ? "" : ",") + backend;
                cardinality += (cardinality.empty() ? "" : ",") + backend + "=" + std::to_string(count);
            }
            add("--only-backends", backends);
            add("--auto-device-counts", cardinality);
            add("--only-strategies", orchestrationStrategyName(modelParityAutomaticStrategy(cell)));
            if (cell.expert_overlay)
                add("--moe-routed-expert-owner-order", cell.expert_overlay->owner_order == RoutedExpertOwnerOrder::Ordinal ? "ordinal" : "random");
        }
        else if (cell.topology.isExpertOverlay())
        {
            const auto &plan = *config.moe_routed_expert_plan;
            if (!plan.placements.empty() || !plan.initial_layer_order_overrides.empty() ||
                plan.continuation_domain_spec.logical_root_participant != 0)
                throw std::invalid_argument("model cell export requires declarative CLI-representable overlay placement");
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
                    throw std::invalid_argument("model cell tier budget must be declarative and exactly expressible in MiB");
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
            add("--mtp-verify-mode", mtpVerifyModeToString(modelParityHttpMTPVerifyMode()));
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
     * @brief Expand remote-host declarations over both mandatory public routes.
     * @param cell Existing selected source; model identity remains in its parent record.
     * @param out Discovery stream, receiving a complete JSON array.
     *
     * A scenario declares intent, never claims that hardware was provisioned or
     * a plan passed. Rank/host binding, remote expert rows and MPI payloads must
     * be authenticated by the later live E2E phase. Local shared-memory traffic
     * cannot satisfy its cross-host obligation.
     */
    inline void writeModelParityCrossHostE2E(const ModelParityCase &cell, std::ostream &out)
    {
        out << '[';
        if (!cell.remote_cpu_overlays.empty())
        {
            const auto args = modelParityServerArguments(
                cell, ModelParityRuntimePlacementProjection::AutomaticRemoteCPUOverlay);
            bool first = true;
            for (const auto &remote : cell.remote_cpu_overlays)
                for (const auto route : kModelParityCrossHostFrontends)
                {
                    const auto route_name = std::string(modelParityCrossHostFrontendName(route));
                    const std::string backend = cell.topology.participants.front().address.isCUDA() ? "cuda" : "rocm";
                    auto route_args = args;
                    route_args.insert(route_args.end(), {"--auto-device-counts",
                        backend + "=1,cpu=" + std::to_string(remote.count())});
                    out << (first ? "" : ",") << "{\"schema\":1,\"id\":"
                        << modelParityJsonString(cell.testName() + "_RemoteCPU" + std::to_string(remote.count()) + "_" + route_name)
                        << ",\"frontend\":" << modelParityJsonString(route_name)
                        << ",\"topology\":{\"kind\":\"cross-host-expert-overlay\",\"continuation_backend\":"
                        << modelParityJsonString(cell.topology.participants.front().address.isCUDA() ? "cuda" : "rocm")
                        << ",\"continuation_devices\":1,\"remote_cpu_hosts\":" << remote.count()
                        << ",\"cpu_ranks_per_host\":1,\"execution_ranks\":" << remote.executionRanks()
                        << ",\"continuation_priority\":0,\"remote_priority\":1}"
                        << ",\"movement_evidence\":\"required\",\"owner_order\":\"ordinal\""
                        << ",\"planning\":{\"mode\":\"auto\",\"strategy\":\"expert-overlay\",\"device_counts\":{"
                        << modelParityJsonString(backend) << ":1,\"cpu\":" << remote.count() << "}}"
                        << ",\"server_policy_args\":[";
                    for (std::size_t i = 0; i < route_args.size(); ++i)
                        out << (i ? "," : "") << modelParityJsonString(route_args[i]);
                    out << "]}";
                    first = false;
                }
        }
        out << ']';
    }

    /**
     * @brief GoogleTest's ADL parameter printer is the discovery export boundary.
     * @param cell Typed parameter, also consumed by the numerical fixture.
     * @param out GoogleTest discovery stream; emits one complete JSON line.
     */
    inline void PrintTo(const ModelParityCase &cell, std::ostream *out)
    {
        // Only the execution policy changes for the serial reference. Its
        // topology, precision, placement and retained MTP capacity stay fixed.
        auto control = cell;
        control.mtp = ModelParityMTP::Off;
        *out << "{\"model_parity_schema\":1,\"id\":" << modelParityJsonString(cell.testName())
             << ",\"model\":" << modelParityJsonString(cell.model.model_path)
             << ",\"runtime\":{\"context_length\":" << cell.model.max_seq_len
             << ",\"generation\":{\"max_tokens\":" << cell.generation_workload.maximumTokens()
             << ",\"minimum_completion_tokens\":" << cell.generation_workload.minimumTokens()
             << ",\"readiness_timeout_seconds\":" << cell.generation_workload.readinessSeconds()
             << ",\"serial_control_id\":" << modelParityJsonString(control.testName())
             << ",\"mtp_policy\":" << modelParityJsonString(modelParityMTPPolicyName(cell.mtp))
             << ",\"mtp_verify_mode\":" << modelParityJsonString(mtpVerifyModeToString(modelParityHttpMTPVerifyMode()))
             << ",\"prefix_state\":" << modelParityJsonString(modelParityPrefixStateName(cell.model.prefix_state))
             << ",\"requests\":[";
        for (std::size_t i = 0; i < ModelParityGenerationWorkload::kProbes.size(); ++i)
        {
            const auto &probe = ModelParityGenerationWorkload::kProbes[i];
            const char *prefix = nullptr;
            switch (probe.prefix)
            {
            case GenerationPrefixProbe::Fresh: prefix = "fresh"; break;
            case GenerationPrefixProbe::FullRestore: prefix = "full"; break;
            case GenerationPrefixProbe::PartialRestore: prefix = "partial"; break;
            }
            *out << (i ? "," : "") << "{\"id\":" << modelParityJsonString(std::string(probe.id))
                 << ",\"prefix\":" << modelParityJsonString(prefix)
                 << ",\"body\":{\"seed\":" << cell.model.generation_seed.value()
                 << ",\"temperature\":0.7,\"top_k\":40,\"top_p\":0.9,"
                    "\"enable_thinking\":false,\"return_token_ids\":true,\"return_runtime_summary\":true,\"max_tokens\":"
                 << cell.generation_workload.maximumTokens()
                 << ",\"messages\":[{\"role\":\"system\",\"content\":"
                 << modelParityJsonString(cell.model.generation_prompt.system())
                 << "},{\"role\":\"user\",\"content\":"
                 << modelParityJsonString(cell.model.generation_prompt.user())
                 << "}";
            // A supplied assistant continuation extends the whole seed prompt,
            // including its generation header. No model answer is spliced into
            // a future request, so MTP and serial requests remain identical.
            if (probe.narrative == GenerationNarrative::Mountain)
            {
                *out << ",{\"role\":\"assistant\",\"content\":"
                     << modelParityJsonString(cell.model.generation_prompt.continuation()) << "}";
                // Public chat closes supplied assistant history. The model may
                // own a next user request, but the observer must still prove
                // that its chat template preserved the entire seed token prefix.
                if (const auto &followup = cell.model.generation_prompt.followup())
                    *out << ",{\"role\":\"user\",\"content\":" << modelParityJsonString(*followup) << "}";
            }
            *out << "]}}";
        }
        *out << "]}"
             << ",\"movement_evidence\":" << modelParityJsonString(modelParityE2EMovementEvidenceName(cell.movementEvidence()))
             << ",\"server_args\":[";
        const auto args = modelParityServerArguments(cell);
        for (std::size_t i = 0; i < args.size(); ++i)
            *out << (i ? "," : "") << modelParityJsonString(args[i]);
        *out << "]},\"cross_host_e2e\":";
        writeModelParityCrossHostE2E(cell, *out);
        *out << ",\"e2e\":";
        if (!cell.e2e_certification) { *out << "null}"; return; }
        const auto &profile = *cell.e2e_certification;
        *out << "{\"context_length\":" << profile.context_length
             << ",\"minimum_prompt_tokens\":" << profile.minimum_prompt_tokens
             << ",\"generation_tokens\":" << profile.generation_tokens
             << ",\"request_timeout_seconds\":" << profile.request_timeout_seconds
             << ",\"readiness_timeout_seconds\":" << profile.readiness_timeout_seconds
             << ",\"cell_timeout_seconds\":{\"AVX512\":" << profile.cell_timeout_seconds.avx512
             << ",\"AVX2\":" << profile.cell_timeout_seconds.avx2 << "}"
             << ",\"thinking_modes\":" << modelParityJsonString(modelParityE2EThinkingModesName(profile.thinking_modes))
             << ",\"movement_evidence\":" << modelParityJsonString(modelParityE2EMovementEvidenceName(cell.movementEvidence()))
             << ",\"planning\":{\"mode\":\"auto\",\"mpi_ranks\":"
             << cell.topology.mpi_ranks << ",\"strategy\":"
             << modelParityJsonString(std::string(orchestrationStrategyName(modelParityAutomaticStrategy(cell))))
             << ",\"device_counts\":{";
        bool first = true;
        for (const auto &[backend, count] : modelParityAutomaticDeviceCounts(cell))
        {
            *out << (first ? "" : ",") << modelParityJsonString(backend) << ':' << count;
            first = false;
        }
        *out << "}";
        if (cell.topology.kind == ModelParityTopologyKind::RankLocalPipelineParallel)
        {
            // This is a shape obligation, not authored placement. Auto retains
            // device ordinals, domain order and layer split as free choices.
            *out << ",\"pipeline_layers\":" << cell.model.transformer_layers
                 << ",\"pipeline_domains\":[";
            const auto sizes = cell.topology.pipeline_stage_sizes.empty()
                ? std::vector<int>(cell.topology.participants.size(), 1) : cell.topology.pipeline_stage_sizes;
            size_t participant = 0;
            for (size_t stage = 0; stage < sizes.size(); ++stage)
            {
                const auto size = sizes[stage];
                const auto &address = cell.topology.participants.at(participant).address;
                *out << (stage ? "," : "") << "{\"backend\":"
                     << modelParityJsonString(address.isCUDA() ? "cuda" : address.isROCm() ? "rocm" : "cpu")
                     << ",\"devices\":" << size << "}";
                participant += size;
            }
            *out << "]";
        }
        *out << "},\"tool_calling\":\"required\",\"server_args\":[";
        const auto automatic_args = modelParityServerArguments(cell, ModelParityRuntimePlacementProjection::AutomaticCell);
        for (std::size_t i = 0; i < automatic_args.size(); ++i)
            *out << (i ? "," : "") << modelParityJsonString(automatic_args[i]);
        *out << "]}}";
    }
} // namespace llaminar2::test::parity
