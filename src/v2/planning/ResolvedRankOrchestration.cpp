/**
 * @file ResolvedRankOrchestration.cpp
 * @brief Shared pure composition of rank compilation and expert authority binding.
 *
 * User topology is copied, resolved against observed hardware, and validated
 * before one result is returned. Simple multi-device MoE is normalized through
 * the same single-tier authority as explicit overlays. No frontend-specific
 * placement, fake model geometry, physical-byte arithmetic or MPI protocol lives
 * here. Runtime and automatic candidates therefore use identical role lowering.
 *
 * Local and global pipeline stages share the original rank-owned prepared
 * source and PMA handoff. Global topology remains immutable metadata; remote
 * execution, activation transport and model-context reuse are not impersonated.
 */
#include "planning/ResolvedRankOrchestration.h"
#include "planning/PlanningModelMetadata.h"
#include "planning/RankMemoryPlanInputs.h"
#include "execution/mpi_orchestration/IExecutionPlanBuilder.h"
#include "execution/mpi_orchestration/ExecutionPlanBuilder.h"
#include "execution/global/NamedDomainGraphBuilder.h"
#include "execution/global_pp/GlobalPPRankPlanBuilder.h"
#include "config/OrchestrationConfigParser.h"
#include "config/OrchestrationStartupPolicy.h"
#include <algorithm>
#include <functional>
#include <stdexcept>
#include <string_view>

namespace llaminar2
{
    namespace
    {
        /**
         * @brief Preserve every validation diagnostic in one fail-closed result.
         * @param errors Complete diagnostics from the existing compiler.
         * @param operation Phase name attached to the rejection.
         * @throws std::invalid_argument when any validation failed.
         */
        void requireValid(const std::vector<std::string> &errors, std::string_view operation)
        {
            if (errors.empty()) return;
            std::string message(operation);
            for (const auto &error : errors) message += "\n  - " + error;
            throw std::invalid_argument(message);
        }

        /**
         * @brief Install one hardware-resolved overlay using shared normalization.
         * @param config Private working copy; the caller's request is never edited.
         * @param inventory Observed rank and device ownership in this namespace.
         * @param origin Explicit declaration or model-driven LocalTP synthesis.
         * @throws std::invalid_argument when binding or normalization fails.
         */
        void bindOverlay(OrchestrationConfig &config, const ClusterInventory &inventory,
                         MoEExpertOverlayPlanInstallOrigin origin)
        {
            if (!config.moe_routed_expert_plan || !config.moe_routed_expert_plan->usesExpertOverlayAuthority())
                return;
            auto bound = bindMoEExpertOverlayPlanToClusterInventory(*config.moe_routed_expert_plan, inventory,
                resolvedMoEOverlayDiagnosticPrefillTransport());
            requireValid(installResolvedMoEExpertOverlayPlan(config, std::move(bound), origin),
                         "Cannot install hardware-resolved ExpertOverlay topology");
        }

        /**
         * @brief Narrow the compiler result to its model-owned overlay role.
         * @param rank_plan Rank/shard facts produced by ExecutionPlanBuilder.
         * @param overlay Bound model topology, never a requested/unresolved map.
         * @param execution Exact continuation/follower ownership for this rank.
         * @throws std::invalid_argument if role and dense-domain geometry disagree.
         */
        void applyOverlayRole(RankExecutionPlan &rank_plan,
                              const MoERoutedExpertPlacementPlan &overlay,
                              const MoEExpertOverlayExecutionPlan &execution)
        {
            // An expert-only endpoint has no ordinary local TP/PP graph. Its
            // device set is owned by the overlay execution plan, not a made-up
            // continuation shard. Preserve the existing CPU control endpoint.
            if (!execution.ownsContinuationGraph())
            {
                rank_plan.local_tp_devices.clear();
                rank_plan.local_tp_weights.clear();
                rank_plan.local_tp_backend = CollectiveBackendType::AUTO;
                rank_plan.local_pp_devices.clear();
                rank_plan.local_pp_layer_boundaries.clear();
                rank_plan.local_pp_stage_tp_info.clear();
                rank_plan.primary_device = GlobalDeviceAddress::cpu();
                rank_plan.weight_shard = {};
                return;
            }
            const auto name = overlay.effectiveBaseModelDomain();
            const auto found = std::find_if(overlay.domains.begin(), overlay.domains.end(),
                [&](const auto &domain) { return domain.name == name; });
            if (found == overlay.domains.end() || found->participants.empty())
                throw std::invalid_argument("ExpertOverlay continuation domain '" + name + "' has no participants");
            const auto &domain = *found;
            if (domain.scope == ExecutionDomainScope::NODE_LOCAL)
            {
                if (overlay.continuation_domain_spec.effectiveDensePolicy() != DenseParallelPolicy::TensorParallel)
                    throw std::invalid_argument("ExpertOverlay NodeTP continuation requires dense_policy=tensor_parallel");
                const bool participates = std::any_of(rank_plan.my_domains.begin(), rank_plan.my_domains.end(),
                    [&](const auto &member) { return member.domain_name == name; });
                if (!participates || !rank_plan.usesGlobalTP() || rank_plan.global_tp_domain_size <= 1)
                    throw std::invalid_argument("ExpertOverlay continuation rank did not retain its cross-rank dense TP domain");
                // NodeTP already owns exact CPU affinity, shard identity and
                // collective membership. Never turn it into another LocalTP.
                rank_plan.local_pp_devices.clear();
                rank_plan.local_pp_layer_boundaries.clear();
                rank_plan.local_pp_stage_tp_info.clear();
                return;
            }
            rank_plan.local_pp_devices.clear();
            rank_plan.local_pp_layer_boundaries.clear();
            rank_plan.local_pp_stage_tp_info.clear();
            rank_plan.primary_device = domain.participants.front();
            rank_plan.local_tp_backend = domain.backend;
            rank_plan.local_tp_weights = domain.weights;
            rank_plan.weight_shard = {};
            if (domain.scope == ExecutionDomainScope::RANK_LOCAL)
            {
                rank_plan.tp_scope = TPScope::RANK_LOCAL;
                rank_plan.local_tp_devices = domain.participants;
            }
            else
            {
                rank_plan.local_tp_devices.clear();
                rank_plan.local_tp_weights.clear();
                rank_plan.tp_scope = TPScope::AUTO;
            }
        }

        /**
         * @brief Validate local stage boundaries against the compiled main model.
         * @param plan Enclosing rank plan, including its real MPI boundary roles.
         * @param model Metadata whose main interval excludes appended predictors.
         * @throws std::invalid_argument for incomplete or foreign stage geometry.
         */
        void requirePipelineGeometry(const RankExecutionPlan &plan, const PlanningModelMetadata &model)
        {
            const auto count = plan.local_pp_devices.size();
            const auto &boundaries = plan.local_pp_layer_boundaries;
            if (count < 2 || boundaries.size() != count + 1 ||
                (!plan.local_pp_stage_tp_info.empty() && plan.local_pp_stage_tp_info.size() != count) ||
                plan.first_layer < 0 || plan.last_layer >= model.mainLayerCount() ||
                boundaries.front() != plan.first_layer || boundaries.back() != plan.last_layer + 1 ||
                std::adjacent_find(boundaries.begin(), boundaries.end(), std::greater_equal<int>{}) != boundaries.end())
                throw std::invalid_argument("ExpertOverlay local pipeline requires exact ordered main-model stage boundaries");
        }

        /**
         * @brief Project one already compiled child without inventing MPI neighbors.
         * @param parent Rank authority retaining the complete authored pipeline.
         * @param scope Exact owned main interval and global component roles.
         * @param participants Inventory-resolved devices of this child alone.
         * @param weights Authored TP workshare of these participants.
         * @param backend Authored collective backend of this child.
         * @return Memory-planning view; local boundary roles are validated by the enclosing pipeline.
         */
        RankExecutionPlan projectPipelineRank(const RankExecutionPlan &parent, const FactoryPPStageConfig &scope,
            const std::vector<GlobalDeviceAddress> &participants, const std::vector<float> &weights,
            CollectiveBackendType backend)
        {
            auto stage = parent;
            stage.first_layer = scope.first_layer;
            stage.last_layer = scope.last_layer - 1;
            stage.has_embedding = scope.has_embedding;
            stage.has_lm_head = scope.has_lm_head;
            stage.primary_device = participants.front();
            stage.local_pp_devices.clear();
            stage.local_pp_layer_boundaries.clear();
            stage.local_pp_stage_tp_info.clear();
            stage.local_pp_backend = CollectiveBackendType::AUTO;
            stage.local_tp_devices = participants;
            stage.local_tp_weights = weights;
            stage.local_tp_backend = backend;
            stage.tp_scope = TPScope::RANK_LOCAL;
            stage.global_tp_domain_id.reset();
            stage.global_tp_rank_in_domain = 0;
            stage.global_tp_domain_size = 1;
            stage.my_domains.clear();
            stage.weight_shard = {};
            return stage;
        }

        /**
         * @brief Project authored routed policies through the exact compiled domain identity.
         * @param parent Original request, before its domain table is retired in the child.
         * @param rank Actual original execution rank owning this stage.
         * @param participants Exact compiler-resolved local participants.
         * @param weights Compiler-resolved workshare, without another physical owner table.
         * @param backend Native collective authority selected by the compiler.
         * @param domain_name Authored name retained by the compiled stage, empty for flat PP.
         * @return Normalizer input; omitted fields retain the existing parent/default policy.
         * @throws std::invalid_argument If a named compiled stage lost its original declaration.
         *
         * Compute resolution remains in ExecutionDomainDefinition, as used by
         * the authority normalizer. This projection only preserves authored
         * typed values before child construction erases parent topology fields.
         */
        MoEExpertOverlayAuthorityPlanRequest pipelineStageAuthorityRequest(
            const OrchestrationConfig &parent, int rank,
            const std::vector<GlobalDeviceAddress> &participants,
            const std::vector<float> &weights, CollectiveBackendType backend,
            const std::string &domain_name)
        {
            MoEExpertOverlayAuthorityPlanRequest request{
                .model_has_routed_experts = true, .world_rank = rank,
                .local_tp_participants = participants, .local_tp_weights = weights,
                .local_tp_backend = backend,
                .routed_compute_policy = parent.routed_expert_compute_policy,
                .owner_order = parent.routed_expert_owner_order,
                .residency_maintenance = parent.moe_rebalance.mode};
            if (domain_name.empty()) return request;
            const DomainDefinition *authored = nullptr;
            for (const auto &domain : parent.domain_definitions)
                if (domain.name == domain_name)
                {
                    if (authored)
                        throw std::invalid_argument("Pipeline stage has duplicate authored execution-domain identity");
                    authored = &domain;
                }
            if (!authored)
                throw std::invalid_argument("Pipeline stage lost its authored execution-domain policy");
            if (authored->routed_compute_policy != RoutedExpertComputePolicy::Unspecified)
                request.routed_compute_policy = authored->routed_compute_policy;
            if (authored->routed_phase_policy != RoutedExpertPhasePolicy::Unspecified)
                request.routed_phase_policy = authored->routed_phase_policy;
            if (authored->routed_decode_assignment_policy != RoutedExpertAssignmentPolicy::Unspecified)
                request.routed_assignment_policy.decode = authored->routed_decode_assignment_policy;
            if (authored->routed_prefill_assignment_policy != RoutedExpertAssignmentPolicy::Unspecified)
                request.routed_assignment_policy.prefill = authored->routed_prefill_assignment_policy;
            return request;
        }

        /**
         * @brief Retire parent placement selectors before binding a child authority.
         * @param parent Immutable serving policy and authored pipeline intent.
         * @param participants Actual child participants, never a topology selector.
         * @param weights Child TP workshare, preserved in participant order.
         * @param backend Exact collective backend selected by the parent compiler.
         * @return Private child configuration retaining model, MTP and cache policy.
         */
        OrchestrationConfig projectPipelineConfig(const OrchestrationConfig &parent,
            const std::vector<GlobalDeviceAddress> &participants, const std::vector<float> &weights,
            CollectiveBackendType backend)
        {
            auto stage = parent;
            stage.device_mode = DeviceAssignmentMode::AUTO;
            stage.device_for_this_rank.reset();
            stage.device_for_this_rank_numa_explicit = false;
            stage.cpu_global_tp_all_local = false;
            stage.device_map.clear();
            stage.device_map_numa_explicit.clear();
            stage.domain_definitions.clear();
            stage.pp_stage_definitions.clear();
            stage.topology_string.clear();
            stage.topology_file_path.clear();
            stage.topology_tree.reset();
            stage.pp_degree = 1;
            stage.cpu_layers = 0;
            stage.cpu_layers_first = false;
            stage.tp_local_degree = 1;
            stage.tp_global_degree = 1;
            stage.tp_degree = static_cast<int>(participants.size());
            stage.tp_scope = TPScope::RANK_LOCAL;
            stage.tp_devices = participants;
            stage.tp_weights = weights;
            stage.default_backend = backend;
            return stage;
        }
    }

    ResolvedRankOrchestration::ResolvedRankOrchestration(
        OrchestrationConfig config, RankExecutionPlan rank_plan,
        std::optional<MoEExpertOverlayExecutionPlan> overlay_execution,
        MoEExpertOverlayAuthorityPlanDisposition origin)
        : config_(std::move(config)), rank_plan_(std::move(rank_plan)),
          overlay_execution_(std::move(overlay_execution)), overlay_origin_(origin) {}

    ResolvedRankOrchestration ResolvedRankOrchestration::resolve(
        const OrchestrationConfig &requested, const PlanningModelMetadata &model,
        const ClusterInventory &inventory, IExecutionPlanBuilder &builder, int rank,
        OrdinaryTerminalProjectionCapacity terminal_projection)
    {
        if (inventory.world_size <= 0 || inventory.ranks.size() != static_cast<size_t>(inventory.world_size) ||
            rank < 0 || rank >= inventory.world_size)
            throw std::invalid_argument("Rank orchestration requires exact inventory membership");
        // Compilation may inspect any peer while resolving a domain. Validate
        // the complete rank namespace, not merely the current rank's record.
        for (int peer = 0; peer < inventory.world_size; ++peer)
            if (inventory.ranks[peer].rank != peer)
                throw std::invalid_argument("Rank orchestration requires exact inventory membership");

        auto config = requested;
        resolveMTPStartupPolicy(config, model.memoryProfile());
        const auto geometry = model.executionModelConfig();
        bindOverlay(config, inventory, MoEExpertOverlayPlanInstallOrigin::UserDeclaredDomains);
        requireValid(builder.validateConfig(config, geometry, inventory), "Configuration validation failed");
        auto rank_plan = builder.buildPlanForRank(config, geometry, inventory, rank);
        // Install authored output demand before immutable child rank projections.
        rank_plan.runtime.ordinary_terminal_projection = terminal_projection;
        if (model.memoryProfile().expert_count > 0 && requiresNamedDomainGlobalGraph(config))
        {
            if (config.moe_routed_expert_plan)
                throw std::invalid_argument("MoE pipeline stages cannot inherit a model-wide expert authority");
            auto *compiler = dynamic_cast<ExecutionPlanBuilder *>(&builder);
            if (!compiler)
                throw std::invalid_argument("Cross-rank MoE pipeline requires its original topology compiler");
            auto global = compiler->buildGlobalPPTopology(config, geometry, inventory);
            requireValid(global.validate(), "Global MoE pipeline topology validation failed");
            if (std::any_of(global.stages.begin(), global.stages.end(), [](const auto &stage) {
                    return stage.inner_mode == InnerParallelism::LOCAL_PP ||
                        (stage.is_global_tp && (stage.per_rank_devices.empty() ||
                            std::any_of(stage.per_rank_devices.begin(),stage.per_rank_devices.end(),
                                [](const auto& device){return !device.isCPU();})));
                }))
                throw std::invalid_argument("MoE prepared global stages require original rank-local devices or a real cross-rank CPU domain");
            config.mtp.depth_defaults_profile = rank_plan.runtime.mtp.depth_defaults_profile;
            config.mtp.terminal_head_policy = rank_plan.runtime.mtp.terminal_head_policy;
            if (rank_plan.runtime.routed_expert_compute_policy == RoutedExpertComputePolicy::Automatic)
                rank_plan.runtime.routed_expert_compute_policy = RoutedExpertComputePolicy::Apportioned;
            const auto actions = GlobalPPRankPlanBuilder::build(global, rank);
            ResolvedRankOrchestration result(config, rank_plan, std::nullopt,
                MoEExpertOverlayAuthorityPlanDisposition::SynthesizedPipeline);
            result.global_pipeline_ = global;
            for (const auto *action : actions.executeStages())
            {
                const auto original_stage=std::find_if(global.stages.begin(),global.stages.end(),
                    [&](const auto& stage){return stage.stage_id==action->stage_id;});
                if(original_stage==global.stages.end() || action->inner_mode==InnerParallelism::LOCAL_PP)
                    throw std::invalid_argument("MoE pipeline lost its original compiled stage");
                const auto& participants=action->is_global_tp?original_stage->per_rank_devices:action->devices;
                const auto& weights=action->is_global_tp?original_stage->tp_weights:action->tp_weights;
                if(participants.empty()) throw std::invalid_argument("MoE pipeline has no original stage participants");
                const FactoryPPStageConfig scope{
                    .first_layer = action->first_layer, .last_layer = action->last_layer + 1,
                    .has_embedding = action->has_embedding, .has_lm_head = action->has_lm_head,
                    .generation_domain = DeviceGenerationExecutionDomain::HostedBoundary};
                auto stage_config = projectPipelineConfig(config, participants, weights, action->backend);
                auto stage_rank = projectPipelineRank(rank_plan, scope, participants, weights, action->backend);
                auto authority_request=pipelineStageAuthorityRequest(config, rank, participants, weights,
                    action->backend,action->domain_name);
                std::optional<ExecutionDomainDefinition> cpu_domain;
                if(action->is_global_tp) {
                    if(original_stage->participating_ranks.size()!=participants.size() ||
                        action->tp_domain_size!=static_cast<int>(participants.size()) ||
                        action->tp_rank_in_domain<0 || action->tp_rank_in_domain>=action->tp_domain_size ||
                        original_stage->participating_ranks[action->tp_rank_in_domain]!=rank)
                        throw std::invalid_argument("CPU pipeline changed its original MPI domain membership");
                    int node=-1;
                    for(std::size_t i=0;i<participants.size();++i) {
                        const int owner=original_stage->participating_ranks[i];
                        if(owner<0 || owner>=inventory.world_size || !participants[i].isCPU())
                            throw std::invalid_argument("CPU pipeline has a foreign original domain rank");
                        const auto& observed=inventory.ranks.at(owner);
                        if(observed.rank!=owner || participants[i].hostname!=observed.hostname ||
                            participants[i].numa_node!=observed.cpu.numa_node || observed.cpu.numa_node<0 ||
                            (node>=0 && observed.node_id!=node))
                            throw std::invalid_argument("CPU pipeline requires its actual original node-local CPU NUMA owners");
                        node=observed.node_id;
                    }
                    cpu_domain.emplace();
                    cpu_domain->name=action->domain_name;
                    cpu_domain->scope=ExecutionDomainScope::NODE_LOCAL;
                    cpu_domain->participants=participants;
                    cpu_domain->weights=weights;
                    cpu_domain->backend=action->backend;
                    cpu_domain->ranks=original_stage->participating_ranks;
                    const auto original=std::find_if(rank_plan.my_domains.begin(),rank_plan.my_domains.end(),
                        [&](const auto& domain){return domain.domain_name==action->domain_name;});
                    if(original==rank_plan.my_domains.end() || original->my_index_in_domain!=action->tp_rank_in_domain)
                        throw std::invalid_argument("CPU pipeline lost its original compiled TP participation");
                    stage_rank.primary_device=action->device;
                    stage_rank.local_tp_devices.clear();stage_rank.local_tp_weights.clear();
                    stage_rank.global_tp_domain_id=original->domain_id;
                    stage_rank.global_tp_rank_in_domain=action->tp_rank_in_domain;
                    stage_rank.global_tp_domain_size=action->tp_domain_size;
                    stage_rank.weight_shard=action->weight_shard;
                    stage_rank.my_domains={*original};stage_rank.tp_scope=TPScope::NODE_LOCAL;
                    stage_config.tp_devices.clear();stage_config.tp_weights.clear();
                    stage_config.tp_scope=TPScope::NODE_LOCAL;
                    stage_config.tp_global_degree=action->tp_domain_size;
                    authority_request.has_cross_rank_tensor_parallel=true;
                }
                // These are stage-local memory claims. Only the retained
                // GlobalPPRankPlan owns inter-rank MPI edges and executes them.
                stage_rank.prev_rank.reset();
                stage_rank.next_rank.reset();
                stage_config.moe_routed_expert_plan = cpu_domain
                    ? normalizeMoEExpertOverlayPipelineStagePlan(authority_request,scope,*cpu_domain)
                    : normalizeMoEExpertOverlayPipelineStagePlan(authority_request,scope);
                // Child runtime consumes the normalized stage policy, including
                // an explicit domain override of the parent scalar setting.
                stage_config.routed_expert_compute_policy =
                    stage_config.moe_routed_expert_plan->domains.front().routed_compute_policy;
                bindOverlay(stage_config, inventory, MoEExpertOverlayPlanInstallOrigin::SynthesizedSimpleTP);
                auto execution = resolveMoEExpertOverlayExecutionPlan(stage_config.moe_routed_expert_plan,
                    {.current_world_rank = rank, .world_size = inventory.world_size});
                applyOverlayRole(stage_rank, *stage_config.moe_routed_expert_plan, execution);
                stage_rank.runtime.routed_expert_compute_policy = stage_config.routed_expert_compute_policy;
                result.pipeline_stages_.push_back(ResolvedMoEPipelineStage(scope, std::move(stage_config),
                    std::move(stage_rank), std::move(execution), action->stage_id));
            }
            if (result.pipeline_stages_.empty())
                throw std::invalid_argument("MoE global pipeline rank owns no declared executable stage");
            return result;
        }
        if (model.memoryProfile().expert_count > 0 && rank_plan.usesLocalPP())
        {
            if (rank_plan.usesGlobalTP() || rank_plan.usesPipelineParallel() || config.moe_routed_expert_plan)
                throw std::invalid_argument("Local MoE pipeline stages cannot inherit a cross-rank or model-wide expert authority");
            requireValid(rank_plan.validate(), "Pipeline plan validation failed");
            requirePipelineGeometry(rank_plan, model);
            config.mtp.depth_defaults_profile = rank_plan.runtime.mtp.depth_defaults_profile;
            config.mtp.terminal_head_policy = rank_plan.runtime.mtp.terminal_head_policy;
            // The parent only coordinates stages. Retain Automatic as authored
            // configuration intent so resolving the result again derives the
            // same per-stage policies; runnable child policies are all concrete.
            if (rank_plan.runtime.routed_expert_compute_policy == RoutedExpertComputePolicy::Automatic)
                rank_plan.runtime.routed_expert_compute_policy = RoutedExpertComputePolicy::Apportioned;
            ResolvedRankOrchestration result(config, rank_plan, std::nullopt,
                MoEExpertOverlayAuthorityPlanDisposition::SynthesizedPipeline);
            for (size_t index = 0; index < rank_plan.local_pp_devices.size(); ++index)
            {
                const FactoryPPStageConfig scope{
                    .first_layer = rank_plan.local_pp_layer_boundaries[index],
                    .last_layer = rank_plan.local_pp_layer_boundaries[index + 1],
                    .has_embedding = rank_plan.has_embedding && index == 0,
                    .has_lm_head = rank_plan.has_lm_head && index + 1 == rank_plan.local_pp_devices.size(),
                    .generation_domain = declaredRankDeviceGenerationDomain(rank_plan)};
                const auto child = rank_plan.local_pp_stage_tp_info.empty()
                    ? RankExecutionPlan::LocalPPStageTPInfo{} : rank_plan.local_pp_stage_tp_info[index];
                const auto participants = child.devices.empty()
                    ? std::vector<GlobalDeviceAddress>{rank_plan.local_pp_devices[index]} : child.devices;
                if (participants.front() != rank_plan.local_pp_devices[index])
                    throw std::invalid_argument("ExpertOverlay pipeline primary device differs from its child TP participants");
                auto stage_config = projectPipelineConfig(config, participants, child.tp_weights, child.tp_backend);
                auto stage_rank = projectPipelineRank(rank_plan, scope, participants, child.tp_weights, child.tp_backend);
                stage_config.moe_routed_expert_plan = normalizeMoEExpertOverlayPipelineStagePlan(
                    pipelineStageAuthorityRequest(config, rank, participants,
                        child.tp_weights, child.tp_backend, child.domain_name), scope);
                // The normalized stage domain is the one concrete policy
                // authority. Parent scalar intent may explicitly differ and
                // must not leak into child runtime/graph preparation.
                stage_config.routed_expert_compute_policy =
                    stage_config.moe_routed_expert_plan->domains.front().routed_compute_policy;
                bindOverlay(stage_config, inventory, MoEExpertOverlayPlanInstallOrigin::SynthesizedSimpleTP);
                auto execution = resolveMoEExpertOverlayExecutionPlan(stage_config.moe_routed_expert_plan,
                    {.current_world_rank = rank, .world_size = inventory.world_size});
                applyOverlayRole(stage_rank, *stage_config.moe_routed_expert_plan, execution);
                stage_rank.runtime.routed_expert_compute_policy = stage_config.routed_expert_compute_policy;
                // RankExecutionPlan::validate() describes an entire MPI rank:
                // no previous peer implies embedding ownership. A local stage
                // instead has the checked parent boundary, never a fake peer.
                result.pipeline_stages_.push_back(ResolvedMoEPipelineStage(scope, std::move(stage_config),
                    std::move(stage_rank), std::move(execution), static_cast<int>(index)));
            }
            return result;
        }
        const auto authority = normalizeMoEExpertOverlayAuthorityPlan({
            .requested_plan = config.moe_routed_expert_plan,
            .model_has_routed_experts = model.memoryProfile().expert_count > 0,
            .world_rank = rank,
            .local_tp_participants = rank_plan.local_tp_devices,
            .local_tp_weights = rank_plan.local_tp_weights,
            .local_tp_backend = rank_plan.local_tp_backend,
            .has_cross_rank_tensor_parallel = rank_plan.usesGlobalTP(),
            .has_pipeline_parallel = rank_plan.usesLocalPP() || rank_plan.usesPipelineParallel(),
            .routed_compute_policy = config.routed_expert_compute_policy,
            .owner_order = config.routed_expert_owner_order,
            .residency_maintenance = config.moe_rebalance.mode,
        });
        config.moe_routed_expert_plan = authority.plan;
        if (authority.synthesized())
        {
            // Synthesis changes the declarative domain inventory. Recompile
            // once before sealing; no consumer may retain the preliminary view.
            bindOverlay(config, inventory, MoEExpertOverlayPlanInstallOrigin::SynthesizedSimpleTP);
            requireValid(builder.validateConfig(config, geometry, inventory), "Implicit ExpertOverlay validation failed");
            rank_plan = builder.buildPlanForRank(config, geometry, inventory, rank);
            rank_plan.runtime.ordinary_terminal_projection = terminal_projection;
        }
        else if (config.routed_expert_compute_policy == RoutedExpertComputePolicy::Automatic)
        {
            // Without an overlay there is no multi-participant expert domain.
            // Seal single-device/dense execution before returning a runnable
            // plan; Automatic is admission intent, never runtime state.
            config.routed_expert_compute_policy = RoutedExpertComputePolicy::Apportioned;
            rank_plan.runtime.routed_expert_compute_policy = config.routed_expert_compute_policy;
        }
        config.mtp.depth_defaults_profile = rank_plan.runtime.mtp.depth_defaults_profile;
        config.mtp.terminal_head_policy = rank_plan.runtime.mtp.terminal_head_policy;
        std::optional<MoEExpertOverlayExecutionPlan> overlay_execution;
        if (config.moe_routed_expert_plan && config.moe_routed_expert_plan->usesExpertOverlayAuthority())
        {
            overlay_execution.emplace(resolveMoEExpertOverlayExecutionPlan(config.moe_routed_expert_plan,
                MoEExpertOverlayExecutionPlanResolverOptions{.current_world_rank = rank, .world_size = inventory.world_size}));
            applyOverlayRole(rank_plan, *config.moe_routed_expert_plan, *overlay_execution);
        }
        requireValid(rank_plan.validate(), "Plan validation failed");
        return ResolvedRankOrchestration(std::move(config), std::move(rank_plan),
                                         std::move(overlay_execution), authority.disposition);
    }
}
