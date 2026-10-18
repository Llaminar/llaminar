/**
 * @file MoEPipelineMemoryAdmission.cpp
 * @brief Compose model-owned stage manifests and complete pipeline memory admission.
 *
 * Shared archive backing and pipeline channels are assembled by the fixed-owner
 * planner once. Expert capacity then joins every contribution through PMA before
 * optional replicas consume remaining capacity. No per-stage free-byte arithmetic
 * or temporary publication of a partially admitted topology is permitted here.
 */
#include "planning/MoEPipelineMemoryAdmission.h"
#include "planning/MoEOverlayPlanningInputs.h"
#include "planning/PlanningModelMetadata.h"
#include "execution/local_execution/engine/PrefillBucketUtils.h"
#include "loaders/ModelLoader.h"
#include <algorithm>
#include <stdexcept>

namespace llaminar2
{
    AdmittedMoEPipelineMemory AdmittedMoEPipelineMemory::admit(
        const ResolvedRankOrchestration &topology, const PlanningModelMetadata &model,
        const ModelLoader &loader, const ClusterInventory &inventory,
        const MoEPipelineMemoryPolicy &policy)
    {
        const auto &profile = model.memoryProfile();
        const auto &parent = topology.rankPlan();
        if (topology.pipelineStages().size() < 2 || topology.overlayExecution() ||
            topology.config().moe_routed_expert_plan || !parent.usesLocalPP() ||
            inventory.world_size != 1 || inventory.ranks.size() != 1 || parent.rank != 0 ||
            inventory.ranks.front().rank != 0 || profile.expert_count <= 0 ||
            profile.architecture != loader.architecture() || profile.n_layers != int(loader.blockCount()) ||
            policy.prefill.bucket_rows.empty() ||
            std::any_of(policy.prefill.bucket_rows.begin(), policy.prefill.bucket_rows.end(),
                [](int rows) { return rows <= 0; }))
            throw std::invalid_argument("MoE pipeline admission requires one complete local topology, source and capture policy");

        AdmittedMoEPipelineMemory result(parent);
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
        for (const auto &child : topology.pipelineStages())
        {
            // Stage storage is reserved before borrowed planner inputs exist.
            // Nonterminal children retain verifier rows but no predictor weights.
            AdmittedMoEPipelineStage stage(child);
            const auto &scope = child.scope();
            const int routed_count = scope.has_lm_head
                ? family.routedLayerCapacity() - scope.first_layer : scope.layerCount();
            stage.family_ = family.forRoutedLayerInterval(scope.first_layer, routed_count);
            stage.manifest_ = buildMoEOverlayLayerWeightManifestFromGGUF(
                loader.getModel(), routed_count, profile.expert_count, scope.first_layer);
            stage.storage_policy_ = resolveMoEOverlayCapacityAdmissionPolicy(
                *child.config().moe_routed_expert_plan, child.config(), inventory.world_size,
                routed_count, profile.expert_count);
            result.stages_.push_back(std::move(stage));
        }

        const std::vector<int> buckets(policy.prefill.bucket_rows.begin(), policy.prefill.bucket_rows.end());
        const auto candidates = segmentedPrefillGraphRowCandidates(buckets,
            parent.runtime.max_seq_len, parent.runtime.moe_routed_prefill.overlay_segment_rows);
        if (candidates.empty())
            throw std::invalid_argument("MoE pipeline admission has no captured-prefill row choices");
        std::string last_capacity_error;
        for (const int rows : candidates)
        {
            try
            {
                std::vector<MoEOverlayLocalCapacityPlannerInput> inputs;
                std::vector<int> segments;
                for (const auto &stage : result.stages_)
                {
                    const auto &child = stage.topology_;
                    auto input = buildMoEOverlayMemoryPlanInputs({
                        .model = profile, .rank_plan = stage.rank_plan_, .config = child.config(),
                        .inventory = inventory, .execution = child.overlayExecution(),
                        .capacity_policy = stage.storage_policy_, .retained_mtp = stage.rank_plan_.runtime.mtp,
                        .graph_family = stage.family_, .prefill = policy.prefill,
                        .gpu_weight_load = policy.gpu_weight_load, .snapshot_capacity = policy.snapshot_capacity,
                        .model_graph_topology_variant_count = policy.model_graph_topology_variant_count,
                        .pipeline_stage = child.scope()}, rows);
                    inputs.push_back(std::move(input.local_capacity));
                    segments.push_back(input.prefill_segment_rows);
                }
                auto fixed = MoEOverlayLocalCapacityPlanner::planPipeline(parent, inputs);
                std::vector<MoEOverlayStageCapacityRequest> requests;
                for (std::size_t index = 0; index < result.stages_.size(); ++index)
                {
                    const auto &stage = result.stages_[index];
                    requests.push_back({*stage.topology_.config().moe_routed_expert_plan,
                        profile.expert_count, stage.manifest_, fixed[index].physical_budgets,
                        stage.storage_policy_});
                }
                auto capacities = MoEOverlayCapacityAdmission::resolvePipelineCapacity(requests);
                if (capacities.size() != result.stages_.size())
                    throw std::logic_error("MoE pipeline capacity did not publish every stage");
                result.admission_ = capacities.front().physical_memory_admission;
                if (!result.admission_)
                    throw std::logic_error("MoE pipeline capacity did not publish one complete transaction");
                for (std::size_t index = 0; index < result.stages_.size(); ++index)
                {
                    if (capacities[index].physical_memory_admission != result.admission_ ||
                        fixed[index].resident_graph_rows != fixed.front().resident_graph_rows ||
                        segments[index] != segments.front())
                        throw std::logic_error("MoE pipeline stages disagree on their aggregate admission or row envelope");
                    auto &stage = result.stages_[index];
                    capacities[index].host_demand_memory = fixed[index].host_demand_memory;
                    stage.capacity_ = std::make_shared<const MoEOverlayResolvedCapacityPlan>(std::move(capacities[index]));
                    stage.devices_ = std::move(fixed[index].device_inputs);
                    stage.rank_plan_.runtime.resident_graph_rows = fixed[index].resident_graph_rows;
                    stage.rank_plan_.runtime.moe_routed_prefill.overlay_segment_rows = segments[index];
                }
                result.rank_plan_.runtime.resident_graph_rows = fixed.front().resident_graph_rows;
                result.rank_plan_.runtime.moe_routed_prefill.overlay_segment_rows = segments.front();
                return result;
            }
            catch (const PhysicalMemoryCapacityExhausted &error)
            {
                last_capacity_error = error.what();
            }
        }
        throw PhysicalMemoryCapacityExhausted("No complete MoE pipeline graph fits: " + last_capacity_error);
    }
}
