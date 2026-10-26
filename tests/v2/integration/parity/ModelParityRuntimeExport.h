/**
 * @file ModelParityRuntimeExport.h
 * @brief Shared production-runtime projection of every canonical model cell.
 *
 * GoogleTest parameter discovery publishes machine-readable metadata without
 * loading weights. Generation regression, tagged E2E and benchmarks consume
 * these records, never test-name fragments or a second model/topology table.
 * Correctness projections retain adversarial policies; benchmark projections
 * retain only cell intent and leave tuning to production defaults.
 * Arguments remain an argv vector
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
     * @brief Preserve routed arithmetic intent while leaving device placement to auto.
     * @param cell Canonical model/topology definition, not a resolved runtime plan.
     * @return The uniform routed policy, or no routed policy for a dense topology.
     * @throws std::invalid_argument when one global auto constraint cannot express
     *         the declared per-domain policies without changing their semantics.
     *
     * Device counts alone do not distinguish whole-expert ownership from movable
     * gate/up ownership. Dropping this expected physical identity would certify
     * the old implementation under a new cell name. Projection launches omit a
     * CLI compute override but retain this identity in their planning evidence;
     * whole-expert A/B controls retain an explicit override. Heterogeneous per-domain policies need an authored
     * placement rather than an invented common policy.
     */
    inline std::optional<RoutedExpertComputePolicy> modelParityAutomaticRoutedComputePolicy(
        const ModelParityCase &cell)
    {
        if (!cell.topology.isExpertOverlay()) return std::nullopt;
        const auto &domains = cell.topology.expert_overlay_plan->domains;
        if (domains.empty()) throw std::invalid_argument("automatic overlay requires routed domains");
        const auto policy = domains.front().routed_compute_policy;
        if (policy == RoutedExpertComputePolicy::Unspecified || policy == RoutedExpertComputePolicy::Automatic ||
            std::any_of(domains.begin(), domains.end(), [policy](const auto &domain) {
                return domain.routed_compute_policy != policy;
            }))
            throw std::invalid_argument("automatic cell requires one explicit routed compute policy");
        return policy;
    }

    /**
     * @brief Share automatic placement intent between HTTP and benchmarks.
     * @param cell Sole model/topology declaration supplying counts and strategy.
     * @return CLI constraints, without performance or correctness overrides.
     * @throws std::invalid_argument for a non-projectable topology.
     *
     * Counts prevent a multi-device cell from quietly measuring one device.
     * Endpoint identities, rank ownership, layer splits and tier capacities
     * remain choices for the real production planner, not this exporter.
     */
    inline std::vector<std::string> modelParityAutomaticPlacementArguments(const ModelParityCase &cell)
    {
        std::string backends, cardinality;
        for (const auto &[backend, count] : modelParityAutomaticDeviceCounts(cell))
        {
            backends += (backends.empty() ? "" : ",") + backend;
            cardinality += (cardinality.empty() ? "" : ",") + backend + "=" + std::to_string(count);
        }
        std::vector<std::string> args{
            "--auto", "--only-backends", backends, "--auto-device-counts", cardinality,
            "--only-strategies", std::string(orchestrationStrategyName(modelParityAutomaticStrategy(cell)))};
        if (cell.expert_overlay)
            args.insert(args.end(), {"--moe-routed-expert-owner-order",
                cell.expert_overlay->owner_order == RoutedExpertOwnerOrder::Ordinal ? "ordinal" : "random"});
        if (const auto policy = modelParityAutomaticRoutedComputePolicy(cell))
        {
            if (*policy == RoutedExpertComputePolicy::GateUpOwnedDownColumns)
            {
                // Projection certification must prove the production default,
                // not force a policy that could hide a broken automatic choice.
                // The expected physical mode remains in the typed declaration
                // and HTTP plan evidence; no second topology rule lives here.
                auto automatic = cell.topology.expert_overlay_plan->domains.front().toExecutionDomainDefinition();
                automatic.routed_compute_policy = RoutedExpertComputePolicy::Automatic;
                if (automatic.resolveRoutedComputePolicy(cell.topology.expert_overlay_plan->routed_tiers.size()) != *policy)
                    throw std::invalid_argument("projection certification requires automatic-default topology eligibility");
            }
            else
            {
                // Whole-expert controls intentionally override the new GPU
                // default. Never relabel that A/B workload as projection mode.
                args.insert(args.end(), {"--moe-routed-expert-compute", routedExpertComputePolicyToString(*policy)});
            }
        }
        return args;
    }

    /**
     * @brief Preserve the selected immutable row-table policy in every runtime projection.
     * @param cell Expanded typed model cell; absent policy leaves ordinary models unchanged.
     * @return Exact explicit storage/window arguments, with no source-path or staging decision.
     */
    inline std::vector<std::string> modelParityNgramStorageArguments(const ModelParityCase &cell)
    {
        if (!cell.ngram_table_policy) return {};
        const auto &policy = *cell.ngram_table_policy;
        policy.validate();
        const auto storage = [&]() -> const char * {
            switch (policy.storage)
            {
            case ImmutableRowStoragePreference::Automatic: return "auto";
            case ImmutableRowStoragePreference::VRAM: return "vram";
            case ImmutableRowStoragePreference::RAM: return "ram";
            case ImmutableRowStoragePreference::MemoryMapped: return "mmap";
            case ImmutableRowStoragePreference::HostAdaptiveRAMSSD: return "host-adaptive-ram-ssd";
            case ImmutableRowStoragePreference::AdaptiveVRAMRAMSSD: return "gpu-adaptive-vram-ram-ssd";
            }
            throw std::invalid_argument("invalid model parity requested ngram storage");
        }();
        std::vector<std::string> arguments{"--ngram-storage", storage,
                "--ngram-mmap-window-bytes", std::to_string(policy.window_bytes),
                "--ngram-mmap-window-count", std::to_string(policy.window_count)};
        if(policy.isFixedHeadSharded())
            arguments.insert(arguments.end(),{"--ngram-distribution","fixed-head-sharded"});
        if(policy.isHostAdaptive())
        {
            const auto &a=policy.host_adaptive;
            arguments.insert(arguments.end(),{"--ngram-cache-max-rows",std::to_string(a.maximum_cache_rows),
                "--ngram-cache-bytes",std::to_string(a.cache_physical_byte_limit),
                "--ngram-cache-aging-accesses",std::to_string(a.policy.aging_accesses),
                "--ngram-cache-protected-numerator",std::to_string(a.policy.protected_numerator),
                "--ngram-cache-protected-denominator",std::to_string(a.policy.protected_denominator)});
        }
        if(policy.isGPUAdaptive())
        {
            const auto &g=policy.gpu_adaptive;const auto &h=g.host;
            arguments.insert(arguments.end(),{
                "--ngram-gpu-cache-max-rows",std::to_string(g.maximum_gpu_cache_rows),
                "--ngram-gpu-cache-bytes",std::to_string(g.gpu_cache_physical_byte_limit),
                "--ngram-gpu-cache-aging-accesses",std::to_string(g.gpu_policy.aging_accesses),
                "--ngram-gpu-cache-protected-numerator",std::to_string(g.gpu_policy.protected_numerator),
                "--ngram-gpu-cache-protected-denominator",std::to_string(g.gpu_policy.protected_denominator),
                "--ngram-source-cache-max-rows",std::to_string(h.maximum_cache_rows),
                "--ngram-source-cache-bytes",std::to_string(h.cache_physical_byte_limit),
                "--ngram-source-cache-aging-accesses",std::to_string(h.policy.aging_accesses),
                "--ngram-source-cache-protected-numerator",std::to_string(h.policy.protected_numerator),
                "--ngram-source-cache-protected-denominator",std::to_string(h.policy.protected_denominator)});
        }
        return arguments;
    }

    namespace model_parity_runtime_detail
    {
        /**
         * @brief Format the original cell's complete declared topology exactly once.
         * @param cell Sole canonical topology/owner-order declaration.
         * @param config Its caller-owned makeOrchestrationConfig result; no resolved/live state.
         * @return Exact public placement argv, without workload or HTTP stress policy.
         * @throws std::invalid_argument For a nondeclarative or nonrepresentable placement.
         * @note Shared by server correctness and diagnostic benchmark exports. This
         *       formatter does not choose participants, reduce quotas or own memory.
         */
        inline std::vector<std::string> declaredPlacementArguments(
            const ModelParityCase &cell, const OrchestrationConfig &config)
        {
            std::vector<std::string> args;
            const auto add = [&](const char *flag, const auto &value)
            {
                std::ostringstream out;
                out << std::setprecision(9) << value;
                args.insert(args.end(), {flag, out.str()});
            };
            if (cell.topology.isExpertOverlay())
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
            return args;
        }

        /** @brief Closed benchmark placement choice; neither arm implies certification. */
        enum class BenchmarkPlacement { AutomaticProduction, DeclaredDiagnostic };
    } // namespace model_parity_runtime_detail

    /**
     * @brief Share benchmark semantic intent without importing HTTP stress policy.
     * @param cell Sole original model/precision/storage/MTP/placement declaration.
     * @param placement Closed automatic-production or declared-diagnostic projection.
     * @return Public argv without executable, model, workload or context.
     * @throws std::invalid_argument For an unknown projection or nonexportable placement.
     */
    inline std::vector<std::string> modelParityBenchmarkSemanticArguments(
        const ModelParityCase &cell, model_parity_runtime_detail::BenchmarkPlacement placement)
    {
        if (placement != model_parity_runtime_detail::BenchmarkPlacement::AutomaticProduction &&
            placement != model_parity_runtime_detail::BenchmarkPlacement::DeclaredDiagnostic)
            throw std::invalid_argument("invalid model parity benchmark placement projection");
        if (placement == model_parity_runtime_detail::BenchmarkPlacement::AutomaticProduction &&
            !cell.e2e_certification)
            throw std::invalid_argument("benchmark projection requires an E2E-tagged case");
        const auto config = cell.makeOrchestrationConfig(cell.model.model_path, 0);
        std::vector<std::string> args;
        if (config.shared_predictor)
            args.insert(args.end(), {"--mtp-model", config.shared_predictor->path()});
        if (placement == model_parity_runtime_detail::BenchmarkPlacement::DeclaredDiagnostic ||
            cell.topology.mpi_ranks > 1)
            args.insert(args.end(), {"--mpi-procs", std::to_string(cell.topology.mpi_ranks)});
        args.insert(args.end(), {"--activation-precision", config.activation_precision,
                                 "--kv-cache-precision", config.kv_cache_precision});
        const auto ngram = modelParityNgramStorageArguments(cell);
        args.insert(args.end(), ngram.begin(), ngram.end());
        if (placement == model_parity_runtime_detail::BenchmarkPlacement::DeclaredDiagnostic)
        {
            args.insert(args.end(), {"--backend", collectiveBackendTypeToString(config.default_backend)});
            if (!config.tp_allreduce_precision_override.empty())
                args.insert(args.end(), {"--tp-allreduce-precision", config.tp_allreduce_precision_override});
            const auto declared_args = model_parity_runtime_detail::declaredPlacementArguments(cell, config);
            args.insert(args.end(), declared_args.begin(), declared_args.end());
        }
        else
        {
            const auto automatic_args = modelParityAutomaticPlacementArguments(cell);
            args.insert(args.end(), automatic_args.begin(), automatic_args.end());
        }
        if (cell.mtpEnabled())
        {
            args.insert(args.end(), {"--mtp", "--mtp-depth-policy",
                                     cell.usesDynamicMTPDepth() ? "dynamic" : "fixed"});
            if (!cell.usesDynamicMTPDepth())
                args.insert(args.end(), {"--mtp-draft-tokens", std::to_string(cell.requestedMTPDraftDepth())});
        }
        else
            args.insert(args.end(), {"--no-mtp", "--mtp-depth-policy", "fixed"});
        if (cell.expert_overlay)
        {
            const bool dynamic = cell.expert_overlay->movement == ModelParityExpertMovement::Dynamic;
            // Maintenance is public automatic-planning intent. A residency
            // declaration instead belongs to an authored placement plan;
            // emitting one here would accidentally construct a partial plan.
            args.insert(args.end(), {"--moe-residency-maintenance", dynamic ? "dynamic" : "off"});
        }
        return args;
    }

    /**
     * @brief Retain the existing E2E-only automatic production benchmark contract.
     * @param cell Existing E2E-tagged canonical model case.
     * @return Existing production-default semantic argv, without workload/model/context.
     * @throws std::invalid_argument For an untagged case; diagnostics cannot grant a tag.
     */
    inline std::vector<std::string> modelParityBenchmarkArguments(const ModelParityCase &cell)
    {
        if (!cell.e2e_certification)
            throw std::invalid_argument("benchmark projection requires an E2E-tagged case");
        return modelParityBenchmarkSemanticArguments(
            cell, model_parity_runtime_detail::BenchmarkPlacement::AutomaticProduction);
    }

    /**
     * @brief Export a diagnostic benchmark preserving the original declared placement.
     * @param cell Existing canonical case; no alternate topology or fake E2E profile.
     * @return Semantic argv using exact declared domains/tiers/stages and production tuning.
     * @throws std::invalid_argument For a nonrepresentable declared placement.
     * @note No HTTP prefix/stress/economic thresholds are copied. This value is
     *       explicitly noncertifying and cannot replace benchmark/e2e eligibility.
     */
    inline std::vector<std::string> modelParityDiagnosticBenchmarkArguments(const ModelParityCase &cell)
    {
        return modelParityBenchmarkSemanticArguments(
            cell, model_parity_runtime_detail::BenchmarkPlacement::DeclaredDiagnostic);
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
        if (config.shared_predictor) add("--mtp-model", config.shared_predictor->path());
        if (declared || cell.topology.mpi_ranks > 1)
            add("--mpi-procs", cell.topology.mpi_ranks);
        add("--activation-precision", config.activation_precision);
        add("--kv-cache-precision", config.kv_cache_precision);
        const auto ngram = modelParityNgramStorageArguments(cell);
        args.insert(args.end(), ngram.begin(), ngram.end());
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
            const auto constraints = modelParityAutomaticPlacementArguments(cell);
            args.insert(args.end(), constraints.begin(), constraints.end());
        }
        else
        {
            const auto declared_args = model_parity_runtime_detail::declaredPlacementArguments(cell, config);
            args.insert(args.end(), declared_args.begin(), declared_args.end());
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
            else
                add("--mtp-depth-policy", "fixed");
        }
        else
            args.insert(args.end(), {"--no-mtp", "--mtp-depth-policy", "fixed"});
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
     * @throws std::invalid_argument if the cell cannot express remote automatic intent.
     *
     * A scenario declares intent, never claims that hardware was provisioned or
     * a plan passed. Rank/host binding, remote expert rows and MPI payloads must
     * be authenticated by the later live E2E phase. Local shared-memory traffic
     * cannot satisfy its cross-host obligation. Topology and planning both use
     * the sealed host-count authority for complete communicator membership.
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
                    // Compute cardinality alone does not seal MPI membership:
                    // the planning contract also needs the continuation rank.
                    // Derive both records from the same typed host declaration.
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
                        << modelParityJsonString(backend) << ":1,\"cpu\":" << remote.count()
                        << "},\"mpi_ranks\":" << remote.executionRanks() << "}"
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
     * @brief Export complete diagnostic obligations using one existing argv owner.
     * @param cell Original expanded case, whose profile was source-validated.
     * @param out Discovery stream; no files, model or approval state are read.
     * @note The typed accuracy policy exports obligations, never observed passes.
     *       Both projections reference diagnostic_benchmark.args. There is no
     *       second argv surface to drift or bypass inventory source remapping.
     */
    inline void writeModelParityPairedWorkload(const ModelParityCase &cell, std::ostream &out)
    {
        if (!cell.paired_workload) return;
        const auto &pair = *cell.paired_workload;
        pair.validate(cell.model.model_path,
            cell.model.shared_predictor ? std::optional<std::string>(cell.model.shared_predictor->path())
                                        : std::nullopt);
        const auto document = [&](const ModelParityDocumentReference &reference)
        {
            out << "{\"path\":" << modelParityJsonString(reference.path)
                << ",\"sha256\":" << modelParityJsonString(reference.sha256) << '}';
        };
        out << ",\"paired_workload\":{\"schema\":2,\"policy\":\"diagnostic_obligation\","
               "\"certification_eligible\":false,\"role\":"
            << modelParityJsonString(pair.role == ModelParityPairRole::Required ? "required" : "expanded")
            << ",\"case_id\":" << modelParityJsonString(cell.testName())
            << ",\"source_fields\":[\"model\"";
        if (cell.model.shared_predictor) out << ",\"shared_predictor\"";
        out << "],\"accuracy\":{\"arguments_from\":\"diagnostic_benchmark\",\"catalog\":";
        document(pair.accuracy_catalog);
        out << ",\"workload\":"; document(pair.accuracy_workload);
        if (const auto *initial = std::get_if<ModelParityInitialAbsolutePilot>(&pair.accuracy_policy))
        {
            out << ",\"policy\":\"initial_absolute_pilot\",\"pilot\":";
            document(initial->pilot);
            out << ",\"approved_baseline\":false,\"calibrated\":false,\"whole_corpus_accepted\":false";
        }
        else
        {
            const auto &revision = std::get<ModelParityCalibratedRevision>(pair.accuracy_policy);
            out << ",\"policy\":\"calibrated_revision\",\"baseline\":";
            document(revision.baseline);
            out << ",\"approval\":"; document(revision.approval);
            out << ",\"calibration\":"; document(revision.calibration);
        }
        out << "},\"benchmark\":{\"policy\":\"diagnostic_declared\","
               "\"arguments_from\":\"diagnostic_benchmark\",\"profile\":";
        document(pair.benchmark_profile);
        out << '}';
        if (pair.qfn_arithmetic)
        {
            const auto &a = *pair.qfn_arithmetic;
            out << ",\"arithmetic\":{\"gdn_projection\":" << modelParityJsonString(std::string(gdnProjectionArithmeticName(a.gdn_projection)))
                << ",\"moe_router\":" << modelParityJsonString(std::string(moeRouterArithmeticName(a.moe_router)))
                << ",\"hc_up\":" << modelParityJsonString(std::string(quantizedActivationOperandPrecisionName(a.hc_up)))
                << ",\"gdn_output\":" << modelParityJsonString(std::string(quantizedActivationOperandPrecisionName(a.gdn_output)))
                << ",\"fa_output\":" << modelParityJsonString(std::string(quantizedActivationOperandPrecisionName(a.fa_output)))
                << ",\"gdn_qkv\":" << modelParityJsonString(std::string(quantizedActivationOperandPrecisionName(a.gdn_qkv))) << '}';
        }
        out << '}';
    }

    /**
     * @brief GoogleTest's ADL parameter printer is the discovery export boundary.
     * @param cell Typed parameter, also consumed by the numerical fixture.
     * @param out GoogleTest discovery stream; emits one complete JSON line.
     */
    inline void PrintTo(const ModelParityCase &cell, std::ostream *out)
    {
        // Reject unknown roles and contradictory runtime eligibility before
        // writing partial metadata. Numerical campaigns remain registered in CTest.
        const auto *inventory_role = modelParityInventoryRoleName(cell.inventory_role);
        if (cell.inventory_role == ModelParityInventoryRole::NumericalDiagnostic &&
            (cell.e2e_certification || !cell.remote_cpu_overlays.empty()))
            throw std::invalid_argument("Numerical diagnostics cannot declare runtime certification eligibility");
        // Reject incomplete direct-constructed cases before writing partial JSON.
        if (cell.paired_workload)
            cell.paired_workload->validate(cell.model.model_path,
                cell.model.shared_predictor ? std::optional<std::string>(cell.model.shared_predictor->path())
                                            : std::nullopt);
        // Only the execution policy changes for the serial reference. Its
        // topology, precision, placement and retained MTP capacity stay fixed.
        auto control = cell;
        control.mtp = ModelParityMTP::Off;
        *out << "{\"model_parity_schema\":1,\"id\":" << modelParityJsonString(cell.testName())
             << ",\"model\":" << modelParityJsonString(cell.model.model_path)
             << (cell.model.shared_predictor ? ",\"shared_predictor\":" +
                 modelParityJsonString(cell.model.shared_predictor->path()) : std::string{});
        // Canonical is the original schema1 default: do not change ordinary
        // configuration bytes or their approved corpus/control identities.
        if (cell.inventory_role != ModelParityInventoryRole::Canonical)
            *out << ",\"inventory_role\":" << modelParityJsonString(inventory_role);
        writeModelParityPairedWorkload(cell, *out);
        *out << ",\"runtime\":{\"context_length\":" << cell.model.max_seq_len
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
        *out << "]},\"diagnostic_benchmark\":{\"schema\":1,\"policy\":\"diagnostic_declared\",\"certification_eligible\":false,\"context_length\":"
             << cell.model.max_seq_len << ",\"args\":[";
        const auto diagnostic_args = modelParityDiagnosticBenchmarkArguments(cell);
        for (std::size_t i = 0; i < diagnostic_args.size(); ++i)
            *out << (i ? "," : "") << modelParityJsonString(diagnostic_args[i]);
        *out << "]},\"cross_host_e2e\":";
        writeModelParityCrossHostE2E(cell, *out);
        *out << ",\"e2e\":";
        if (!cell.e2e_certification) { *out << "null,\"benchmark\":null}"; return; }
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
        if (const auto policy = modelParityAutomaticRoutedComputePolicy(cell))
            *out << ",\"routed_compute\":" << modelParityJsonString(routedExpertComputePolicyToString(*policy));
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
        *out << "]},\"benchmark\":{\"schema\":1,\"policy\":\"production_defaults\",\"context_length\":"
             << profile.context_length << ",\"args\":[";
        const auto benchmark_args = modelParityBenchmarkArguments(cell);
        for (std::size_t i = 0; i < benchmark_args.size(); ++i)
            *out << (i ? "," : "") << modelParityJsonString(benchmark_args[i]);
        *out << "]}}";
    }
} // namespace llaminar2::test::parity
