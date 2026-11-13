/**
 * @file MoEPipelineMemoryAdmission.cpp
 * @brief Compose model-owned stage manifests and complete pipeline memory admission.
 *
 * Shared archive backing and pipeline channels are assembled by the fixed-owner
 * planner once. Expert capacity then joins every contribution through PMA before
 * optional replicas consume remaining capacity. No per-stage free-byte arithmetic
 * or temporary publication of a partially admitted topology is permitted here.
 * CPU serial storage declarations survive admission and runtime projection.
 *
 * Local and global pipeline stages share the original rank-owned prepared
 * source and PMA handoff. Global topology remains immutable metadata; remote
 * execution, activation transport and model-context reuse are not impersonated.
 */
#include "planning/MoEPipelineMemoryAdmission.h"
#include "planning/OriginalRankCapacityExchange.h"
#include "execution/runner/RankInitializationLifecycle.h"
#include "execution/mpi_orchestration/ExecutionPlanBuilder.h"
#include <map>
#include "execution/moe/MoEOverlayNoPeerResidency.h"
#include "backends/NativeContextPreparation.h"
#include "planning/MoEOverlayPlanningInputs.h"
#include "planning/PlanningModelMetadata.h"
#include "planning/PlanningHostServiceEvidence.h"
#include "execution/local_execution/engine/PrefillBucketUtils.h"
#include "loaders/ModelLoader.h"
#include <algorithm>
#include <stdexcept>

namespace llaminar2
{
    namespace
    {
        /**
         * @brief Bind one original cold source into a runtime copy without changing topology.
         * @param plan Parent or child runtime projection, still before fixed/optional planning.
         * @param source Exact serving source; null preserves source-free discovery only.
         * @throws std::logic_error For an absent/replaced inherited owner or an unfrozen/foreign source.
         */
        void bindOriginalPipelineSource(RankExecutionPlan& plan,
            const std::shared_ptr<NativeContextPreparation>& source)
        {
            const auto& inherited=plan.runtime.native_context_preparation;
            if(inherited && (!source || inherited.get()!=source.get() ||
                inherited.owner_before(source) || source.owner_before(inherited)))
                throw std::logic_error("Pipeline admission cannot omit or replace its original native source");
            if(source){
                if(source->memory()->worldRank()!=plan.rank ||
                    source->memory()->phase()!=PhysicalMemoryAuthorityPhase::Preparing)
                    throw std::logic_error("Pipeline admission requires its original Preparing rank source");
                (void)source->mandatoryBOM(); // Requires the original closed planning snapshot.
            }
            plan.runtime.native_context_preparation=source;
        }
    }
    void AdmittedMoEPipelineMemory::publishRuntimeCapacity(RuntimeConfig &runtime) const
    {
        const auto& original=rank_plan_.runtime.native_context_preparation;
        const auto& current=runtime.native_context_preparation;
        if(current && (!original || current.get()!=original.get() ||
            current.owner_before(original) || original.owner_before(current)))
            throw std::logic_error("Pipeline capacity publication cannot replace its original native source");
        runtime.native_context_preparation=original;
        runtime.resident_graph_rows = rank_plan_.runtime.resident_graph_rows;
        runtime.moe_routed_prefill.overlay_segment_rows = rank_plan_.runtime.moe_routed_prefill.overlay_segment_rows;
        runtime.ngram_table_placement = rank_plan_.runtime.ngram_table_placement;
        runtime.attention_participants = rank_plan_.runtime.attention_participants;
        runtime.cpu_serial_expert_storage = rank_plan_.runtime.cpu_serial_expert_storage;
        runtime.retained_host_services = rank_plan_.runtime.retained_host_services;
    }

    AdmittedMoEPipelineMemory AdmittedMoEPipelineMemory::admit(
        const ResolvedRankOrchestration &topology, const PlanningModelMetadata &model,
        const ModelLoader &loader, const ClusterInventory &inventory,
        const MoEPipelineMemoryPolicy &policy)
    {
        const auto &profile = model.memoryProfile();
        auto parent = topology.rankPlan();
        const bool global = topology.globalPipeline().has_value();
        if (topology.pipelineStages().empty() || (!global && topology.pipelineStages().size() < 2) ||
            topology.overlayExecution() || topology.config().moe_routed_expert_plan ||
            (!global && (!parent.usesLocalPP() || inventory.world_size != 1 || parent.rank != 0)) || inventory.world_size <= 0 ||
            inventory.ranks.size() != static_cast<size_t>(inventory.world_size) ||
            parent.rank < 0 || parent.rank >= inventory.world_size ||
            inventory.ranks[parent.rank].rank != parent.rank ||
            (global && topology.globalPipeline()->world_size != inventory.world_size) || profile.expert_count <= 0 ||
            profile.architecture != loader.architecture() || profile.n_layers != int(loader.blockCount()) ||
            policy.prefill.bucket_rows.empty() ||
            std::any_of(policy.prefill.bucket_rows.begin(), policy.prefill.bucket_rows.end(),
                [](int rows) { return rows <= 0; }))
            throw std::invalid_argument("MoE pipeline admission requires one complete rank-owned topology, source and capture policy");

        for (int rank = 0; rank < inventory.world_size; ++rank)
            if (inventory.ranks[rank].rank != rank)
                throw std::invalid_argument("MoE pipeline admission requires exact observed rank membership");
        // The immutable topology predates serving-context preparation. Carry
        // the actual already-created owner through runtime copies before the
        // existing shared fixed-BOM planner deduplicates it across all stages.
        bindOriginalPipelineSource(parent,policy.native_context_preparation);
        AdmittedMoEPipelineMemory result(parent);
        result.global_pipeline_ = topology.globalPipeline();
        result.stages_.reserve(topology.pipelineStages().size());
        const auto &mtp = parent.runtime.mtp;
        const int decode_rows = retainsMTPGraphCapacity(mtp) ? resolveMTPRetainedTargetQueryRows(mtp) : 1;
        const auto family = resolveMoEOverlayInferenceGraphFamilyIdentity(loader,
            loader.architecture(), loader.blockCount(), retainsMTPGraphCapacity(mtp)
                ? MoEOverlayMTPGraphFamilyPolicy::RetainModelSidecars : MoEOverlayMTPGraphFamilyPolicy::MainOnly,
            1, std::max(decode_rows, parent.runtime.moe_routed_prefill.overlay_segment_rows),
            decode_rows, parent.runtime.batch_size, resolveMTPRetainedDraftCapacity(mtp));
        if (family.main_layer_count != model.mainLayerCount())
            throw std::invalid_argument("MoE pipeline descriptor differs from the model's main-layer manifest");
        const auto declaration=[&](const ResolvedMoEPipelineStage& child,bool own_source) {
            AdmittedMoEPipelineStage stage(child);
            if(own_source)bindOriginalPipelineSource(stage.rank_plan_,policy.native_context_preparation);
            const auto &scope = child.scope();
            const int routed_count = scope.has_lm_head
                ? family.routedLayerCapacity() - scope.first_layer : scope.layerCount();
            stage.family_ = family.forRoutedLayerInterval(scope.first_layer, routed_count);
            stage.manifest_ = buildMoEOverlayLayerWeightManifestFromLoader(loader, routed_count, profile.expert_count, scope.first_layer);
            stage.storage_policy_ = resolveMoEOverlayCapacityAdmissionPolicy(
                *child.config().moe_routed_expert_plan, child.config(), inventory.world_size,
                routed_count, profile.expert_count);
            return stage;
        };

        // All stage policies are already authored-domain normalized here. A
        // Static tail still consumes the minimum when an upstream controller
        // is Dynamic; parent defaults cannot replace these original policies.
        std::optional<DeviceType> native_type;
        bool homogeneous_local = !global;
        bool active_upstream = false;
        const auto observe_native = [&](const GlobalDeviceAddress &device) {
            if (!native_type) native_type = device.device_type;
            homogeneous_local &= device.toLocalDeviceId().is_gpu() && device.device_type == *native_type;
        };
        for (const auto &device : parent.local_pp_devices) observe_native(device);
        for (const auto &stage : parent.local_pp_stage_tp_info) for (const auto &device : stage.devices) observe_native(device);
        for (const auto &child : topology.pipelineStages()) {
            active_upstream |= !child.scope().has_lm_head && child.config().moe_rebalance.mode == MoERebalanceRuntimeMode::Dynamic &&
                child.config().moe_routed_expert_plan && !MoEOverlayNoPeerResidency::hasNoPeerTopology(*child.config().moe_routed_expert_plan);
        }
        const bool native_maintenance_budget = homogeneous_local && active_upstream && mtp.enabled;

        const std::vector<int> buckets(policy.prefill.bucket_rows.begin(), policy.prefill.bucket_rows.end());
        const auto candidates = segmentedPrefillGraphRowCandidates(buckets,
            parent.runtime.max_seq_len, parent.runtime.moe_routed_prefill.overlay_segment_rows);
        if (candidates.empty())
            throw std::invalid_argument("MoE pipeline admission has no captured-prefill row choices");
        std::optional<MoEOverlayCapacityDraft> selected;
        const auto cpu_programs=topology.globalPipeline()
            ? originalCPUPipelinePrograms(topology.config(),*topology.globalPipeline(),inventory)
            : std::vector<OriginalCPUPipelineProgram>{};
        const bool cross_rank=global && !cpu_programs.empty() && inventory.world_size>1;
        const auto& original=policy.original_parent;
        if(cross_rank && original && (original->rank()!=parent.rank || original->world_size()!=inventory.world_size ||
            original->communicator()==MPI_COMM_NULL))
            throw std::invalid_argument("Pipeline capacity requires its same original parent rank/context");
        if(cross_rank && !original && policy.native_context_preparation)
            throw std::invalid_argument("Serving pipeline capacity cannot replace original-rank evidence with source-free planning");
        const auto capacity_phase=[&](std::uint32_t ordinal,std::string_view name,auto&& work) {
            auto phase=RankInitializationLifecycle::execute({ordinal,name},std::forward<decltype(work)>(work),
                [&](auto identity,auto outcome)->RankInitializationConsensusResult {
                    if(!original)return {.outcome=outcome==RankInitializationLocalOutcome::Succeeded?
                        RankInitializationConsensusOutcome::AllRanksSucceeded:RankInitializationConsensusOutcome::AtLeastOneRankFailed};
                    return MPIRankInitializationConsensus::reach(original->communicator(),identity,outcome);
                });
            switch(phase.status) {
                case RankInitializationPhaseStatus::Succeeded:
                case RankInitializationPhaseStatus::LocalStepReturnedFailure:
                case RankInitializationPhaseStatus::LocalStepThrewException:
                case RankInitializationPhaseStatus::PeerStepFailed:return phase;
                default:throw std::runtime_error(phase.diagnostic("Original pipeline capacity"));
            }
        };
        std::map<int,ResolvedRankOrchestration> compiled;
        std::map<int,std::vector<AdmittedMoEPipelineStage>> owned;
        std::vector<AdmittedMoEPipelineStage> canonical;
        const auto declare_original=[&] {
            for(const auto& child:topology.pipelineStages())result.stages_.push_back(declaration(child,true));
            if(!cross_rank)return true;
            compiled.emplace(parent.rank,topology);owned.emplace(parent.rank,result.stages_);
            for(int rank=0;rank<inventory.world_size;++rank)if(rank!=parent.rank) {
                ExecutionPlanBuilder compiler;
                auto foreign=ResolvedRankOrchestration::resolve(topology.config(),model,inventory,compiler,rank);
                const auto entry=compiled.emplace(rank,std::move(foreign)).first;
                std::vector<AdmittedMoEPipelineStage> stages;
                for(const auto& child:entry->second.pipelineStages())stages.push_back(declaration(child,false));
                owned.emplace(rank,std::move(stages));
            }
            // Metadata comes from the SAME original declaration/model/inventory.
            // Only actual local native/fixed contributions are exchanged below;
            // these foreign projections never create contexts or grant capacity.
            for(const auto& stage:topology.globalPipeline()->stages) {
                const int owner=stage.is_global_tp?stage.participating_ranks.front():stage.owning_rank;
                const auto& declarations=owned.at(owner);
                const auto found=std::find_if(declarations.begin(),declarations.end(),
                    [&](const auto& child){return child.topology_.stageId()==stage.stage_id;});
                if(found==declarations.end() || found->topology_.scope().first_layer!=stage.first_layer ||
                    found->topology_.scope().last_layer!=stage.last_layer+1)
                    throw std::invalid_argument("Pipeline capacity omitted an original semantic stage declaration");
                canonical.push_back(*found);
            }
            return true;
        };
        if(cross_rank && original) {
            const auto ready=capacity_phase(0,"pipeline-capacity.original-declarations",declare_original);
            if(!ready.succeeded())throw std::invalid_argument(ready.diagnostic("Original pipeline capacity declarations"));
            requireIdenticalOverlayGraphRowCandidates(candidates,original);
        } else (void)declare_original();
        std::vector<MoEOverlayLocalCapacityPlannerResult> selected_fixed;
        std::vector<int> selected_segments;
        std::map<int,std::vector<std::vector<DevicePlanConfig>>> selected_source_free_groups;
        std::string last_capacity_error;
        auto prefill_policy = policy.prefill;
        prefill_policy.exact_fixed_pipeline_widths = !global && parent.runtime.ngram_table_policy.isFixedHeadSharded();
        const auto fixed_for=[&](const ResolvedRankOrchestration& owner_topology,
            const std::vector<AdmittedMoEPipelineStage>& declarations,int rows) {
            std::vector<MoEOverlayLocalCapacityPlannerInput> inputs;
            std::vector<int> segments;
            for (const auto &stage : declarations)
            {
                const auto &child = stage.topology_;
                const auto original_cpu=std::find_if(cpu_programs.begin(),cpu_programs.end(),
                    [&](const auto& program){return program.stageId()==child.stageId();});
                auto input = buildMoEOverlayMemoryPlanInputs({
                    .model = profile, .rank_plan = stage.rank_plan_, .config = child.config(),
                    .inventory = inventory, .execution = child.overlayExecution(),
                    .capacity_policy = stage.storage_policy_, .retained_mtp = stage.rank_plan_.runtime.mtp,
                    .graph_family = stage.family_, .prefill = prefill_policy,
                    .gpu_weight_load = policy.gpu_weight_load, .snapshot_capacity = policy.snapshot_capacity,
                    .model_graph_topology_variant_count = policy.model_graph_topology_variant_count,
                    .pipeline_stage = child.scope(), .original_expert_manifest=stage.manifest_,
                    .weight_precision=WeightPrecision::NATIVE,
                    .native_source_placement=loader.nativeWeightSourcePlacement(),
                    .native_pipeline_maintenance_budget=native_maintenance_budget,
                    .original_cpu_stage_program=original_cpu==cpu_programs.end()?nullptr:&*original_cpu}, rows);
                input.local_capacity.original_cpu_pipeline_programs=cpu_programs;
                input.local_capacity.native_ordering_events=policy.native_ordering_events;
                input.local_capacity.retained_host_services=policy.retained_host_services;
                inputs.push_back(std::move(input.local_capacity));
                segments.push_back(input.prefill_segment_rows);
            }
            auto fixed = MoEOverlayLocalCapacityPlanner::planPipeline(owner_topology, inputs);
            return std::pair{std::move(fixed),std::move(segments)};
        };
        for(const int rows:candidates) {
            std::vector<MoEOverlayLocalCapacityPlannerResult> fixed;
            std::vector<int> segments;
            std::optional<OriginalPipelineCapacitySendFrame> frame;
            std::map<int,std::vector<std::vector<DevicePlanConfig>>> source_free_groups;
            const auto local_work=[&] {
                auto candidate=fixed_for(topology,result.stages_,rows);
                fixed=std::move(candidate.first);segments=std::move(candidate.second);
                if(cross_rank && original) {
                    std::vector<OriginalPipelineFixedContribution> contributions;
                    for(size_t index=0;index<result.stages_.size();++index)
                        contributions.push_back({result.stages_[index].topology_.stageId(),fixed[index].physical_budgets});
                    frame=OriginalPipelineCapacitySendFrame::prepare(*topology.globalPipeline(),parent.rank,contributions);
                }
                return true;
            };
            const auto complete_work=[&] {
                std::vector<OriginalPipelineStageBudgets> complete;
                if(cross_rank && original) complete=gatherOriginalPipelineCapacityBudgets(*frame,original,*topology.globalPipeline());
                else if(cross_rank) {
                    for(const auto& stage:canonical)complete.push_back({stage.topology_.stageId(),{}});
                    // Source-free discovery reuses the SAME fixed planner on
                    // every immutable original rank projection. It performs no
                    // native birth/admission and never supplies serving evidence.
                    for(int rank=0;rank<inventory.world_size;++rank) {
                        auto candidate=rank==parent.rank?std::pair{fixed,segments}:fixed_for(compiled.at(rank),owned.at(rank),rows);
                        for(size_t index=0;index<owned.at(rank).size();++index) {
                            const int stage=owned.at(rank)[index].topology_.stageId();
                            auto target=std::find_if(complete.begin(),complete.end(),[&](const auto& entry){return entry.stage_id==stage;});
                            if(target==complete.end())throw std::logic_error("Source-free planning produced a foreign original stage");
                            target->budgets.insert(target->budgets.end(),candidate.first[index].physical_budgets.begin(),candidate.first[index].physical_budgets.end());
                            if(rank!=parent.rank)source_free_groups[rank].push_back(std::move(candidate.first[index].device_inputs));
                        }
                    }
                }
                std::vector<MoEOverlayStageCapacityRequest> requests;
                const auto& declarations=cross_rank?canonical:result.stages_;
                for(size_t index=0;index<declarations.size();++index) {
                    const auto& stage=declarations[index];
                    if(cross_rank && (index>=complete.size() || complete[index].stage_id!=stage.topology_.stageId()))
                        throw std::logic_error("Pipeline capacity exchange changed original stage order");
                    requests.push_back({*stage.topology_.config().moe_routed_expert_plan,profile.expert_count,stage.manifest_,
                        cross_rank?complete[index].budgets:fixed[index].physical_budgets,stage.storage_policy_});
                }
                // The stage plans are authored config plans, not overlay-bound
                // copies: admission observes the uniform CPU layout from the
                // SAME live inventory (with the same cross-layout refusal) the
                // hardware binder would use, so a PP stage with a CPU expert
                // tier is admitted instead of refused as an unbound plan.
                selected=MoEOverlayCapacityAdmission::resolvePipelineCapacityDraft(
                    requests, &inventory);
                return true;
            };
            if(cross_rank && original) {
                if (auto rejected = runOriginalPipelineCapacityTrial(*original, rows,
                        "pipeline-capacity.local-fixed", "Original pipeline fixed capacity", local_work)) {
                    last_capacity_error = std::move(*rejected);
                    continue;
                }
                if (auto rejected = runOriginalPipelineCapacityTrial(*original, rows,
                        "pipeline-capacity.complete-fixed", "Original pipeline complete capacity", complete_work)) {
                    last_capacity_error = std::move(*rejected);
                    selected.reset();
                    continue;
                }
            } else {
                try {(void)local_work();(void)complete_work();}
                catch(const PhysicalMemoryCapacityExhausted& error){selected.reset();last_capacity_error=error.what();continue;}
            }
            selected_fixed=std::move(fixed);selected_segments=std::move(segments);
            selected_source_free_groups=std::move(source_free_groups);break;
        }
        if(!selected)
            throw PhysicalMemoryCapacityExhausted("No complete MoE pipeline graph fits: "+last_capacity_error);
        std::vector<ImmutableRowTableAdmissionRequest> requests;
        const auto build_table_requests=[&] {
            std::vector<std::vector<DevicePlanConfig>> groups;
            for(const auto &stage:selected_fixed)groups.push_back(stage.device_inputs);
            const auto upload=MemoryPlanner::immutableRowTableUploadBindings(profile,groups);
            const auto request=MemoryPlanner::immutableRowTableAdmissionRequest(profile,parent.runtime.ngram_table_policy,
                groups,upload,ImmutableRowTableHostPlacement::fromObservedNUMANode(parent.numa_node));
            if(request)requests.push_back(*request);
            for(const auto& [rank,rank_groups]:selected_source_free_groups) {
                const auto bindings=MemoryPlanner::immutableRowTableUploadBindings(profile,rank_groups);
                const auto extra=MemoryPlanner::immutableRowTableAdmissionRequest(profile,parent.runtime.ngram_table_policy,
                    rank_groups,bindings,ImmutableRowTableHostPlacement::fromObservedNUMANode(compiled.at(rank).rankPlan().numa_node));
                if(extra)requests.push_back(*extra);
            }
        };
        // Table rejection is outside the row and replica searches. No earlier
        // physical authority exists, and this failure cannot choose fewer rows.
        std::optional<ImmutableRowTableAdmissionSendFrame> table_frame;
        if(cross_rank && original) {
            const auto ready=capacity_phase(0,"pipeline-capacity.final-table-frame",[&] {
                build_table_requests();
                table_frame=ImmutableRowTableAdmissionSendFrame::prepare(requests,parent.rank,&profile);
                return true;
            });
            if(!ready.succeeded())throw std::invalid_argument(ready.diagnostic("Original pipeline final table metadata"));
            requests=gatherOverlayImmutableTableRequests(*table_frame,original,profile);
        } else build_table_requests();
        auto capacities=selected->admit(requests);
        if(cross_rank) {
            if(capacities.size()!=canonical.size())throw std::logic_error("Pipeline capacity omitted an original global stage");
            std::vector<MoEOverlayResolvedCapacityPlan> local;
            for(const auto& stage:result.stages_) {
                const auto found=std::find_if(canonical.begin(),canonical.end(),
                    [&](const auto& entry){return entry.topology_.stageId()==stage.topology_.stageId();});
                if(found==canonical.end())throw std::logic_error("Pipeline capacity cannot project a foreign local stage");
                local.push_back(std::move(capacities[static_cast<size_t>(found-canonical.begin())]));
            }
            capacities=std::move(local);
        }
        if(capacities.size()!=result.stages_.size())
            throw std::logic_error("MoE pipeline finalization lost its exact stage count");
        for(const auto &placement:capacities.front().ngram_table_placements)
            if(placement->worldRank()==parent.rank)
            {
                result.rank_plan_.runtime.ngram_table_placement=placement;
                for(auto &stage:result.stages_)stage.rank_plan_.runtime.ngram_table_placement=placement;
            }
        result.admission_ = capacities.front().physical_memory_admission;
        if (!result.admission_)
            throw std::logic_error("MoE pipeline capacity did not publish one complete transaction");
        for (std::size_t index = 0; index < result.stages_.size(); ++index)
        {
            if (capacities[index].physical_memory_admission != result.admission_ ||
                selected_fixed[index].resident_graph_rows != selected_fixed.front().resident_graph_rows ||
                selected_segments[index] != selected_segments.front())
                throw std::logic_error("MoE pipeline stages disagree on their aggregate admission or row envelope");
            auto &stage = result.stages_[index];
            capacities[index].host_demand_memory = selected_fixed[index].host_demand_memory;
            capacities[index].activation_batch_inventory = selected_fixed[index].activation_batch_inventory;
            capacities[index].physical_gpu_lane_demand = selected_fixed[index].physical_gpu_lane_demand;
            capacities[index].cpu_serial_expert_storage =
                MemoryPlanner::cpuSerialExpertStorageAdmissions(selected_fixed[index].device_inputs);
            stage.rank_plan_.runtime.cpu_serial_expert_storage = capacities[index].cpu_serial_expert_storage;
            stage.capacity_ = std::make_shared<const MoEOverlayResolvedCapacityPlan>(std::move(capacities[index]));
            stage.devices_ = std::move(selected_fixed[index].device_inputs);
            stage.rank_plan_.runtime.retained_host_services = retainedHostServicesFromInputs(
                stage.devices_, stage.rank_plan_.rank);
            stage.rank_plan_.runtime.resident_graph_rows = selected_fixed[index].resident_graph_rows;
            stage.rank_plan_.runtime.moe_routed_prefill.overlay_segment_rows = selected_segments[index];
        }
        result.rank_plan_.runtime.retained_host_services = result.stages_.front().rank_plan_.runtime.retained_host_services;
        for (const auto& stage : result.stages_)
            if (stage.rank_plan_.runtime.retained_host_services != result.rank_plan_.runtime.retained_host_services)
                throw std::logic_error("Pipeline stages lost their single original CPU service declaration session");
        result.rank_plan_.runtime.resident_graph_rows = selected_fixed.front().resident_graph_rows;
        result.rank_plan_.runtime.moe_routed_prefill.overlay_segment_rows = selected_segments.front();
        std::vector<DevicePlanConfig> admitted_attention_inputs;
        for (auto &stage : result.stages_)
        {
            stage.rank_plan_.runtime.attention_participants =
                projectAdmittedQwenSparseAttentionPrograms(
                    profile, stage.devices_, stage.rank_plan_.rank);
            admitted_attention_inputs.insert(admitted_attention_inputs.end(),
                                             stage.devices_.begin(),
                                             stage.devices_.end());
        }
        result.rank_plan_.runtime.attention_participants =
            projectAdmittedQwenSparseAttentionPrograms(
                profile, admitted_attention_inputs, result.rank_plan_.rank);
        return result;
    }
}
