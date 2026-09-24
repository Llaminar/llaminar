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
#include "planning/ResolvedRankOrchestration.h"
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
        std::shared_ptr<const MoEOverlayResolvedCapacityPlan> overlay)
        : candidate_(std::move(candidate)), ranks_(std::move(ranks)), devices_(std::move(devices)),
          admission_(std::move(admission)), overlay_(std::move(overlay))
    {
        if (!admission_ || ranks_.size() != candidate_.membership.discoveryRanks().size())
            throw std::logic_error("Candidate admission did not seal its complete rank/physical plan");
        // All rank references in the compiled configuration are already in
        // this compact namespace. Saving only that configuration would lose
        // which discovery processes must actually construct its runners.
        candidate_.config.execution_rank_selection = candidate_.membership.selection();
        candidate_.config.mpi_procs = candidate_.membership.discoverySize();
    }

    AdmittedOrchestrationCandidate AdmittedOrchestrationCandidate::admit(
        AutomaticOrchestrationCandidate candidate, const PlanningModelSource &source,
        const OrchestrationCandidateMemoryPolicy &policy)
    {
        if (candidate.config.model_path != source.path())
            throw std::invalid_argument("Candidate model path does not name the retained planning source");
        if (!std::holds_alternative<ApplyOrchestrationRequest>(resolveOrchestrationIntent(candidate.config)))
            throw std::invalid_argument("Candidate admission requires explicit placement, not a second automatic search");
        if (policy.prefill_bucket_rows.empty() ||
            std::any_of(policy.prefill_bucket_rows.begin(), policy.prefill_bucket_rows.end(), [](int rows) { return rows <= 0; }))
            throw std::invalid_argument("Candidate admission requires explicit positive captured-prefill buckets");
        const auto &model = source.metadata();
        const auto &profile = model.memoryProfile();
        const auto &loader = source.loader();
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
            auto certificate = std::make_shared<const PhysicalMemoryPlanAdmissionCertificate>(memory.memory_plan.admit());
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
        const auto manifest = buildMoEOverlayLayerWeightManifestFromGGUF(loader.getModel(), family.routedLayerCapacity(), profile.expert_count);
        const auto capacity_policy = resolveMoEOverlayCapacityAdmissionPolicy(
            overlay_plan, config, inventory.world_size, family.routedLayerCapacity(), profile.expert_count);
        const auto rows = segmentedPrefillGraphRowCandidates(policy.prefill_bucket_rows,
            runtime.max_seq_len, runtime.moe_routed_prefill.overlay_segment_rows);
        if (rows.empty()) throw std::invalid_argument("Candidate overlay has no captured-prefill row choices");
        std::string last_capacity_error;
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
                auto capacity = MoEOverlayCapacityAdmission::resolveCapacity(
                    overlay_plan, profile.expert_count, manifest, budgets, capacity_policy);
                for (size_t index = 0; index < ranks.size(); ++index)
                {
                    ranks[index].runtime.resident_graph_rows = local[index].resident_graph_rows;
                    ranks[index].runtime.moe_routed_prefill.overlay_segment_rows = segments[index];
                }
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
                std::vector<DevicePlanConfig> devices;
                for (auto &rank : local)
                    devices.insert(devices.end(), std::make_move_iterator(rank.device_inputs.begin()),
                        std::make_move_iterator(rank.device_inputs.end()));
                auto certificate = capacity.physical_memory_admission;
                return {std::move(candidate), std::move(ranks), std::move(devices), std::move(certificate),
                    std::make_shared<const MoEOverlayResolvedCapacityPlan>(std::move(capacity))};
            }
            catch (const PhysicalMemoryCapacityExhausted &error)
            {
                // This is a legal shape search, not recovery from invalid
                // lowering. No other exception can select a smaller graph.
                last_capacity_error = error.what();
            }
        }
        throw PhysicalMemoryCapacityExhausted("No complete ExpertOverlay candidate graph fits: " + last_capacity_error);
    }
}
