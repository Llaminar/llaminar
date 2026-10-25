/**
 * @file OrchestrationCandidateAdmission.cpp
 * @brief Canonical compiler/BOM composition for a complete candidate namespace.
 *
 * Every rank is priced on the root's immutable observation without constructing
 * remote runners or entering MPI collectives. Runtime uses the same input and
 * capacity builders after live resource initialization. Only typed capacity
 * exhaustion permits another graph-row candidate; invalid geometry, overflow
 * and unsupported codebooks remain their original errors.
 */
#include "planning/OrchestrationCandidateAdmission.h"
#include "execution/mtp/MTPWeightManifest.h"
#include "planning/ResolvedRankOrchestration.h"
#include "config/OrchestrationStartupPolicy.h"
#include "planning/RankMemoryPlanInputs.h"
#include "planning/MoEOverlayMemoryPlanInputs.h"
#include "planning/MoEOverlayPlanningInputs.h"
#include "execution/mpi_orchestration/ExecutionPlanBuilder.h"
#include "execution/local_execution/engine/PrefillBucketUtils.h"
#include "loaders/ModelLoader.h"
#include "utils/DebugEnv.h"
#include "utils/PrefillGraphBucketDefaults.h"
#include <algorithm>
#include <iterator>
#include <stdexcept>

namespace llaminar2
{
    namespace
    {
        /** @brief Project final per-rank table inputs from already frozen graph geometry.
         * @param profile Complete original source directory and immutable model metadata.
         * @param ranks Compiled execution ranks carrying the lossless requested policy.
         * @param devices Winning higher-priority participant inputs, never a new row candidate.
         * @return At most one canonical table request per rank with actual PLE consumers.
         */
        std::vector<ImmutableRowTableAdmissionRequest> candidateTableRequests(
            const ModelMemoryProfile &profile,const std::vector<RankExecutionPlan> &ranks,
            const std::vector<DevicePlanConfig> &devices)
        {
            std::vector<ImmutableRowTableAdmissionRequest> requests;
            for(const auto &rank:ranks)
            {
                std::vector<DevicePlanConfig> local;
                for(const auto &device:devices)if(device.world_rank==rank.rank)local.push_back(device);
                const std::vector<std::vector<DevicePlanConfig>> groups{std::move(local)};
                const auto upload=MemoryPlanner::immutableRowTableUploadBindings(profile,groups);
                auto request=MemoryPlanner::immutableRowTableAdmissionRequest(profile,
                    rank.runtime.ngram_table_policy,groups,upload,ImmutableRowTableHostPlacement::fromObservedNUMANode(rank.numa_node));
                if(request)requests.push_back(std::move(*request));
            }
            return requests;
        }

        /** @brief Publish the same sealed placement into its compiled rank and exact consumers.
         * @param placements Final canonical selections from this candidate's sole certificate.
         * @param ranks Mutable setup publication into returned runtime configurations.
         * @param devices Frozen participant geometry; only placement metadata is added.
         * @throws std::logic_error For a placement without its compiled rank or consumer.
         */
        void publishCandidateTables(
            const std::vector<std::shared_ptr<const ResolvedImmutableRowTablePlacement>> &placements,
            std::vector<RankExecutionPlan> &ranks,std::vector<DevicePlanConfig> &devices)
        {
            for(const auto &placement:placements)
            {
                const auto rank=std::find_if(ranks.begin(),ranks.end(),[&](const auto &r) {
                    return placement && r.rank==placement->worldRank();
                });
                if(rank==ranks.end())throw std::logic_error("Final candidate table has no exact compiled rank");
                rank->runtime.ngram_table_placement=placement;
                for(const auto &consumer:placement->consumers())
                {
                    bool found=false;
                    for(auto &device:devices)
                        if(device.world_rank==consumer.world_rank && device.device==consumer.device)
                        {device.ngram_table_placement=placement;found=true;}
                    if(!found)throw std::logic_error("Final candidate table has no exact compiled PLE consumer");
                }
            }
        }
    }

    OrchestrationCandidateMemoryPolicy OrchestrationCandidateMemoryPolicy::fromStartup()
    {
        const auto &env = debugEnv();
        return {.prefill_bucket_rows = env.execution.prefill_graph_bucket_sizes,
            .minimum_prefill_sequence_rows = env.execution.prefill_graph_min_seq,
            .maximum_cached_prefill_buckets = env.execution.prefill_graph_max_cached_buckets,
            .weight_load = configuredGPUWeightLoadMemoryPolicy()};
    }

    AdmittedOrchestrationCandidate::AdmittedOrchestrationCandidate(
        AutomaticOrchestrationCandidate candidate, std::vector<RankExecutionPlan> ranks,
        std::vector<DevicePlanConfig> devices,
        std::shared_ptr<const PhysicalMemoryPlanAdmissionCertificate> admission,
        std::shared_ptr<const MoEOverlayResolvedCapacityPlan> overlay,
        std::shared_ptr<const AdmittedMoEPipelineMemory> pipeline)
        : candidate_(std::move(candidate)), ranks_(std::move(ranks)), devices_(std::move(devices)),
          admission_(std::move(admission)), overlay_(std::move(overlay)), pipeline_(std::move(pipeline))
    {
        if (!admission_ || ranks_.size() != candidate_.membership.discoveryRanks().size())
            throw std::logic_error("Candidate admission did not seal its complete rank/physical plan");
        if (pipeline_ && (overlay_ || pipeline_->physicalAdmission() != admission_))
            throw std::logic_error("Candidate pipeline admission has conflicting expert or physical authorities");
        // All rank references in the compiled configuration are already in
        // this compact namespace. Saving only that configuration would lose
        // which discovery processes must actually construct its runners.
        candidate_.config.execution_rank_selection = candidate_.membership.selection();
        candidate_.config.mpi_procs = candidate_.membership.discoverySize();
    }

    std::optional<AdmittedMoELayerView> AdmittedOrchestrationCandidate::expertLayer(int layer) const
    {
        const auto owns = [layer](const auto &capacity) {
            return std::any_of(capacity.layer_footprints.begin(), capacity.layer_footprints.end(),
                [layer](const auto &entry) { return entry.layer_idx == layer; });
        };
        if (pipeline_)
        {
            for (const auto &stage : pipeline_->stages())
                if (owns(*stage.capacity()))
                    return AdmittedMoELayerView{*stage.topology().config().moe_routed_expert_plan, *stage.capacity()};
        }
        else if (overlay_)
        {
            if (owns(*overlay_)) return AdmittedMoELayerView{*candidate_.config.moe_routed_expert_plan, *overlay_};
        }
        else return std::nullopt;
        throw std::invalid_argument("Candidate has no admitted expert authority for global layer " + std::to_string(layer));
    }

    AdmittedOrchestrationCandidate AdmittedOrchestrationCandidate::admit(
        AutomaticOrchestrationCandidate candidate, const PlanningModelSource &source,
        const OrchestrationCandidateMemoryPolicy &policy)
    {
        if (candidate.config.model_path != source.path() || candidate.config.shared_predictor != source.predictorSource())
            throw std::invalid_argument("Candidate model path does not name the retained planning source");
        if (!std::holds_alternative<ApplyOrchestrationRequest>(resolveOrchestrationIntent(candidate.config)))
            throw std::invalid_argument("Candidate admission requires explicit placement, not a second automatic search");
        if (policy.prefill_bucket_rows.empty() ||
            std::any_of(policy.prefill_bucket_rows.begin(), policy.prefill_bucket_rows.end(), [](int rows) { return rows <= 0; }))
            throw std::invalid_argument("Candidate admission requires explicit positive captured-prefill buckets");
        const auto &model = source.metadata();
        const auto &profile = model.memoryProfile();
        const auto &loader = source.loader();
        resolveMTPStartupPolicy(candidate.config, profile);
        // Direct callers of candidate admission must obey the same learned-
        // predictor contract as automatic startup, before publishing a BOM.
        if (retainsMTPGraphCapacity(candidate.config.mtp))
            (void)requireMTPWeightManifest(loader, loader.architecture(),
                                           static_cast<int>(loader.blockCount()));
        const auto &inventory = candidate.membership.inventory();
        const auto weight_load = resolveGPUWeightLoadMemoryGeometry(
            maximumGGUFTensorPayloadBytes(loader.getModel()), policy.weight_load);

        ExecutionPlanBuilder builder;
        std::vector<ResolvedRankOrchestration> compiled;
        std::vector<RankExecutionPlan> ranks;
        compiled.reserve(inventory.ranks.size());
        ranks.reserve(inventory.ranks.size());
        for (int rank = 0; rank < inventory.world_size; ++rank)
        {
            compiled.push_back(ResolvedRankOrchestration::resolve(candidate.config, model, inventory, builder, rank));
            ranks.push_back(compiled.back().rankPlan());
        }
        // The shared compiler has already bound the complete topology. Keep its
        // normalized configuration rather than the unbound proposal or a second
        // frontend reconstruction of continuation and expert-only roles.
        candidate.config = compiled.front().config();
        const auto &config = candidate.config;
        if (!compiled.front().pipelineStages().empty())
        {
            if (compiled.size() != 1)
                throw std::logic_error("MoE local pipeline admission requires its exact process-local rank namespace");
            if (std::any_of(compiled.front().pipelineStages().begin(), compiled.front().pipelineStages().end(),
                    [](const auto &stage) {
                        const auto &devices = stage.overlayExecution().currentRankPlan().local_devices;
                        return std::any_of(devices.begin(), devices.end(), [](DeviceId device) { return device.is_gpu(); });
                    }) && (!debugEnv().execution.gpu_graphs || !debugEnv().execution.prefill_graph_buckets))
                throw std::logic_error("Automatic GPU pipeline admission requires production captured generation");
            auto pipeline = std::make_shared<const AdmittedMoEPipelineMemory>(AdmittedMoEPipelineMemory::admit(
                compiled.front(), model, loader, inventory, {
                    .prefill = {.bucket_rows = policy.prefill_bucket_rows,
                        .minimum_sequence_rows = policy.minimum_prefill_sequence_rows,
                        .maximum_cached_buckets = policy.maximum_cached_prefill_buckets},
                    .gpu_weight_load = {.policy = policy.weight_load, .maximum_source_bytes = weight_load.maximum_source_bytes}}));
            std::vector<DevicePlanConfig> devices;
            for (const auto &stage : pipeline->stages())
                devices.insert(devices.end(), stage.devicePlans().begin(), stage.devicePlans().end());
            ranks.front() = pipeline->rankPlan();
            auto certificate = pipeline->physicalAdmission();
            return {std::move(candidate), std::move(ranks), std::move(devices),
                std::move(certificate), nullptr, std::move(pipeline)};
        }
        const bool overlay = compiled.front().overlayExecution().has_value();
        if (std::any_of(compiled.begin(), compiled.end(), [&](const auto &rank) {
                return rank.overlayExecution().has_value() != overlay;
            })) throw std::logic_error("Candidate ranks disagree on their expert authority");

        if (!overlay)
        {
            std::vector<DevicePlanConfig> devices;
            for (const auto &rank : ranks)
            {
                auto local = buildRankMemoryPlanInputs({
                    .model = profile, .plan = rank, .inventory = inventory.ranks.at(rank.rank),
                    .weight_load_geometry = weight_load,
                    .captured_prefill_buckets = policy.prefill_bucket_rows,
                    .max_gpu_memory_bytes = config.memoryLimitBytes(DeviceType::CUDA),
                    .max_cpu_memory_bytes = config.memoryLimitBytes(DeviceType::CPU)});
                devices.insert(devices.end(), std::make_move_iterator(local.begin()), std::make_move_iterator(local.end()));
            }
            const bool gpu = std::any_of(devices.begin(), devices.end(), [](const auto &device) { return device.device.is_gpu(); });
            if (gpu && (!debugEnv().execution.gpu_graphs || !debugEnv().execution.prefill_graph_buckets))
                throw std::logic_error("Automatic GPU admission requires production captured generation");
            // The same selector owns CPU full-context and GPU bucket geometry;
            // do not duplicate its interpretation of an unset activation row cap.
            auto memory = MemoryPlanner::planLargestFittingResidentGraphRows(profile, devices,
                normalizePrefillGraphBuckets(policy.prefill_bucket_rows));
            // Rows are frozen before the lowest-priority table is considered.
            // A final table failure escapes directly; it cannot rerun this search.
            auto final_plan=memory.memory_plan.physicalPlan();
            std::vector<std::shared_ptr<const ResolvedImmutableRowTablePlacement>> placements;
            for(const auto &request:candidateTableRequests(profile,ranks,memory.device_inputs))
            {
                auto selected=ImmutableRowTablePlacementPlanner::select(request,final_plan);
                placements.push_back(std::move(selected.placement));
                final_plan=std::move(selected.physical_plan);
            }
            auto certificate=std::make_shared<const PhysicalMemoryPlanAdmissionCertificate>(final_plan);
            publishCandidateTables(placements,ranks,memory.device_inputs);
            for (auto &rank : ranks)
                rank.runtime.attention_participants = projectAdmittedQwenSparseAttentionPrograms(
                    profile, memory.device_inputs, rank.rank);
            if (memory.resident_graph_rows <= 0)
                throw std::logic_error("Candidate admission returned no executable graph rows");
            for (auto &rank : ranks) rank.runtime.resident_graph_rows = memory.resident_graph_rows;
            // The memory selector preserves ordinary CPU continuation capacity
            // even when a GPU peer uses a smaller bucket. Return its exact
            // evaluated inputs instead of repeating that policy here.
            return {std::move(candidate), std::move(ranks), std::move(memory.device_inputs),
                std::move(certificate), nullptr};
        }

        const auto &overlay_plan = *config.moe_routed_expert_plan;
        if (std::any_of(overlay_plan.domains.begin(), overlay_plan.domains.end(), [](const auto &domain) {
                return std::any_of(domain.participants.begin(), domain.participants.end(), [](const auto &device) { return device.isGPU(); });
            }) && (!debugEnv().execution.gpu_graphs || !debugEnv().execution.prefill_graph_buckets))
            throw std::logic_error("Automatic GPU overlay admission requires production captured generation");
        const auto &runtime = ranks.front().runtime;
        const auto &mtp = runtime.mtp;
        const int decode_rows = retainsMTPGraphCapacity(mtp) ? std::max(1, resolveMTPRetainedTargetQueryRows(mtp)) : 1;
        const auto family = resolveMoEOverlayInferenceGraphFamilyIdentity(loader, loader.architecture(), loader.blockCount(),
            retainsMTPGraphCapacity(mtp) ? MoEOverlayMTPGraphFamilyPolicy::RetainModelSidecars : MoEOverlayMTPGraphFamilyPolicy::MainOnly,
            1, std::max(decode_rows, std::max(1, runtime.moe_routed_prefill.overlay_segment_rows)),
            decode_rows, std::max(1, runtime.batch_size), resolveMTPRetainedDraftCapacity(mtp));
        const auto manifest = buildMoEOverlayLayerWeightManifestFromLoader(loader, family.routedLayerCapacity(), profile.expert_count);
        const auto capacity_policy = resolveMoEOverlayCapacityAdmissionPolicy(
            overlay_plan, config, inventory.world_size, family.routedLayerCapacity(), profile.expert_count);
        const auto rows = segmentedPrefillGraphRowCandidates(policy.prefill_bucket_rows,
            runtime.max_seq_len, runtime.moe_routed_prefill.overlay_segment_rows);
        if (rows.empty()) throw std::invalid_argument("Candidate overlay has no captured-prefill row choices");
        std::string last_capacity_error;
        std::optional<MoEOverlayCapacityDraft> selected_draft;
        std::vector<MoEOverlayLocalCapacityPlannerResult> selected_local;
        std::vector<int> selected_segments;
        for (const int candidate_rows : rows)
        {
            try
            {
                std::vector<MoEOverlayBoundPhysicalMemoryBudget> budgets;
                std::vector<MoEOverlayLocalCapacityPlannerResult> local;
                std::vector<int> segments;
                for (size_t index = 0; index < ranks.size(); ++index)
                {
                    const auto inputs = buildMoEOverlayMemoryPlanInputs({
                        .model = profile, .rank_plan = ranks[index], .config = config,
                        .inventory = inventory, .execution = *compiled[index].overlayExecution(),
                        .capacity_policy = capacity_policy, .retained_mtp = mtp, .graph_family = family,
                        .prefill = {.bucket_rows = policy.prefill_bucket_rows,
                            .minimum_sequence_rows = policy.minimum_prefill_sequence_rows,
                            .maximum_cached_buckets = policy.maximum_cached_prefill_buckets},
                        .gpu_weight_load = {.policy = policy.weight_load, .maximum_source_bytes = weight_load.maximum_source_bytes}},
                        candidate_rows);
                    local.push_back(MoEOverlayLocalCapacityPlanner::plan(inputs.local_capacity));
                    segments.push_back(inputs.prefill_segment_rows);
                    budgets.insert(budgets.end(), local.back().physical_budgets.begin(), local.back().physical_budgets.end());
                }
                auto draft=MoEOverlayCapacityAdmission::resolveCapacityDraft(
                    overlay_plan,profile.expert_count,manifest,budgets,capacity_policy);
                selected_draft.emplace(std::move(draft));
                selected_local=std::move(local);
                selected_segments=std::move(segments);
                break;
            }
            catch (const PhysicalMemoryCapacityExhausted &error)
            {
                // Only the higher-priority row/quota draft may choose a smaller shape.
                last_capacity_error=error.what();
            }
        }
        if(!selected_draft)
            throw PhysicalMemoryCapacityExhausted("No complete ExpertOverlay candidate graph fits: " + last_capacity_error);
        for(size_t index=0;index<ranks.size();++index)
        {
            ranks[index].runtime.resident_graph_rows=selected_local[index].resident_graph_rows;
            ranks[index].runtime.moe_routed_prefill.overlay_segment_rows=selected_segments[index];
        }
        std::vector<DevicePlanConfig> devices;
        for(auto &rank:selected_local)
            devices.insert(devices.end(),std::make_move_iterator(rank.device_inputs.begin()),
                std::make_move_iterator(rank.device_inputs.end()));
        // This extension is outside every row/quota catch. No incomplete PMA
        // certificate exists and a table rejection cannot change the frozen draft.
        const auto requests=candidateTableRequests(profile,ranks,devices);
        auto capacities=selected_draft->admit(requests);
        if(capacities.size()!=1u)
            throw std::logic_error("Candidate overlay final admission lost its exact stage");
        auto capacity=std::move(capacities.front());
        publishCandidateTables(capacity.ngram_table_placements,ranks,devices);
        for (auto &rank : ranks)
            rank.runtime.attention_participants =
                projectAdmittedQwenSparseAttentionPrograms(profile, devices, rank.rank);
        /*
         * Auto's capacity result proves and ranks this topology at
         * discovery time; it is not a user-authored fixed quota.
         * Backend contexts and rank-local graph resources are created
         * after that observation. Persisting every last estimated
         * expert slot as FixedPerLayer would turn a small, legitimate
         * change in free VRAM into a startup failure even though the
         * same automatic policy fits with one fewer resident expert.
         * Preserve auto quota intent in the apply document. The live
         * runner resolves its complete BOM against its refreshed
         * PhysicalMemoryAuthority observation before allocating any
         * expert, without changing topology or movement policy.
         */
        auto certificate=capacity.physical_memory_admission;
        return {std::move(candidate),std::move(ranks),std::move(devices),std::move(certificate),
            std::make_shared<const MoEOverlayResolvedCapacityPlan>(std::move(capacity))};
    }
}
