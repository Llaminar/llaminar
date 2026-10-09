/**
 * @file PreparedWeightAuthorityIdentity.cpp
 * @brief Structural identity shared by initial admission and prepared-context reuse.
 *
 * Stage and model-wide consumers use the same ordered policy serialization.
 * Only metadata is inspected; model and cache payloads are never hashed. Active
 * request controls may change within the unchanged admitted capacity envelope.
 */
#include "PreparedWeightAuthorityIdentity.h"
#include "planning/MoEPipelineMemoryAdmission.h"
#include "execution/local_execution/engine/PrefillBucketUtils.h"
#include "utils/DebugEnv.h"
#include <sstream>

namespace llaminar2
{
    bool retainedMTPWeightAuthorityMatches(const MTPRuntimeConfig &retained, const MTPRuntimeConfig &requested)
    {
        return resolveMTPRetainedDraftCapacity(retained) == resolveMTPRetainedDraftCapacity(requested) &&
            retained.max_request_batch == requested.max_request_batch &&
            retained.sidecar_dense_policy == requested.sidecar_dense_policy &&
            retained.terminal_head_policy == requested.terminal_head_policy;
    }

    /**
     * @brief Serialize the exact requested topology that changes prepared weights.
     *
     * This is a collision-free process-local identity, not a display string
     * or a probabilistic hash. Length-prefixing every string and spelling
     * every ordered field lets a retained ModelContext reject a changed
     * participant, shard policy, owner order, tier quota, or explicit
     * placement before memory admission credits existing device payloads.
     * Capacity-affecting controller policy and normalized retained-graph
     * geometry are included because automatic tier filling consumes the
     * bytes left after those fixed owners. The active cycles-per-wave cap
     * is deliberately excluded: it can use fewer already-materialized
     * lanes without changing any prepared byte. Model-aware resolved
     * quotas are carried separately by the reuse contract and are never
     * re-solved against VRAM that already contains the retained weights.
     */
    std::string routedWeightAuthorityRequestIdentity(
        const std::shared_ptr<MoERoutedExpertPlacementPlan> &plan,
        const OrchestrationConfig &config)
    {
        if (!plan)
            return {};

        std::ostringstream out;
        const auto append_string = [&out](const std::string &value)
        {
            out << value.size() << ':' << value << ';';
        };
        const auto append_ints = [&out](const auto &values)
        {
            out << values.size() << '[';
            for (const auto value : values)
                out << value << ',';
            out << ']';
        };
        const auto append_floats = [&out](const auto &values)
        {
            out << values.size() << '[' << std::hexfloat;
            for (const float value : values)
                out << value << ',';
            out << std::defaultfloat << ']';
        };

        out << "enabled=" << plan->enabled
            << ";topology=" << static_cast<int>(plan->topology)
            << ";residency="
            << static_cast<int>(plan->residency_policy)
            << ";owner_order=" << static_cast<int>(plan->owner_order)
            << ';';
        append_string(plan->continuation_domain);
        append_string(plan->base_model_domain);
        append_string(plan->shared_expert_domain);

        const auto &continuation = plan->continuation_domain_spec;
        append_string(continuation.domain);
        out << continuation.logical_root_participant << ','
            << continuation.dense_tp_enabled << ','
            << continuation.dense_decode_replicated << ','
            << continuation.dense_decode_mirrored_embedding << ','
            << static_cast<int>(continuation.dense_policy) << ','
            << static_cast<int>(continuation.hidden_layout) << ','
            << continuation.shared_expert_uses_dense_tp << ';';

        out << "dense=" << plan->dense_domains.size() << '{';
        for (const auto &domain : plan->dense_domains)
            append_string(domain.toString());
        out << "};routed=" << plan->domains.size() << '{';
        for (const auto &domain : plan->domains)
        {
            append_string(domain.name);
            out << static_cast<int>(domain.scope) << ','
                << static_cast<int>(domain.backend) << ','
                << domain.owner_rank << ','
                << static_cast<int>(domain.routed_compute_policy) << ','
                << static_cast<int>(domain.routed_phase_policy) << ','
                << static_cast<int>(domain.routed_decode_assignment_policy)
                << ','
                << static_cast<int>(domain.routed_prefill_assignment_policy)
                << ';';
            out << domain.participants.size() << '[';
            for (const auto &participant : domain.participants)
                append_string(participant.toString());
            out << ']';
            append_ints(domain.world_ranks);
            append_floats(domain.weights);
        }
        out << "};tiers=" << plan->routed_tiers.size() << '{';
        for (const auto &tier : plan->routed_tiers)
        {
            append_string(tier.name);
            append_string(tier.domain);
            out << tier.priority << ','
                << tier.max_experts_per_layer << ','
                << tier.memory_budget_bytes << ','
                << tier.fallback << ';';
        }
        out << "};initial_order_overrides="
            << plan->initial_layer_order_overrides.size() << '{';
        for (const auto &order : plan->initial_layer_order_overrides)
        {
            out << order.layer << ':';
            append_ints(order.expert_ids);
        }
        out << "};placements=" << plan->placements.size() << '{';
        for (const auto &placement : plan->placements)
        {
            out << placement.layer << ':';
            append_ints(placement.routed_expert_tier);
        }
        out << '}';

        const auto graph_candidates =
            segmentedPrefillGraphRowCandidates(
                debugEnv().execution.prefill_graph_bucket_sizes,
                config.max_seq_len,
                config.moe_routed_prefill.overlay_segment_rows);
        const int requested_graph_rows = graph_candidates.empty()
                                             ? 0
                                             : resolveRetainedGraphRowCapacity(
                                                   graph_candidates.front(),
                                                   config.mtp);
        const auto append_optional_bytes = [&out](const auto &value)
        {
            if (value)
                out << *value;
            else
                out << "auto";
            out << ',';
        };
        out << ";capacity="
            << static_cast<int>(config.moe_rebalance.mode) << ','
            << config.moe_rebalance.migration_transfer_slots << ','
            << config.moe_rebalance.dynamic_max_swaps_per_layer << ','
            // Both values affect the admitted evidence banks, even when
            // expert quotas and retained graph geometry are unchanged.
            << config.moe_rebalance.window_size << ','
            << config.moe_rebalance.max_window_size << ','
            << requested_graph_rows << ','
            << resolveMTPRetainedDraftCapacity(config.mtp) << ','
            << config.mtp.max_request_batch << ','
            << static_cast<int>(config.mtp.sidecar_dense_policy) << ','
            << static_cast<int>(config.mtp.terminal_head_policy) << ','
            << config.batch_size << ','
            << config.max_seq_len << ','
            << debugEnv().execution.prefill_graph_max_cached_buckets
            << ',';
        append_optional_bytes(config.max_gpu_memory_mb);
        append_optional_bytes(config.max_cpu_memory_mb);
        append_string(config.activation_precision);
        append_string(config.kv_cache_precision);
        return out.str();
    }

    namespace
    {
        /** @brief Append one global interval and its complete capacity-affecting policy. */
        void appendPipelineStage(std::ostream &out, const ResolvedMoEPipelineStage &stage)
        {
            const auto &scope = stage.scope();
            const auto &config = stage.config();
            const auto identity = routedWeightAuthorityRequestIdentity(config.moe_routed_expert_plan, config);
            const auto &cache = config.prefix_cache;
            out << scope.first_layer << ',' << scope.last_layer << ',' << scope.has_embedding << ',' << scope.has_lm_head
                << '/' << identity.size() << ':' << identity << '/'
                << cache.enabled << ',' << static_cast<int>(cache.storage_mode) << ',' << cache.block_size << ','
                << cache.ram_budget_bytes << ',' << cache.device_budget_bytes << ',' << cache.disk_budget_bytes << ','
                << static_cast<int>(cache.terminal_state) << ',' << static_cast<int>(cache.moe_policy)
                << '/' << cache.disk_dir.size() << ':' << cache.disk_dir << ';';
        }
    }

    std::string pipelineRoutedWeightAuthorityRequestIdentity(const ResolvedRankOrchestration &topology)
    {
        std::ostringstream out;
        out << "pipeline-expert-authority-v1/" << topology.pipelineStages().size() << '/';
        for (const auto &stage : topology.pipelineStages()) appendPipelineStage(out, stage);
        return out.str();
    }

    std::string pipelineRoutedWeightAuthorityRequestIdentity(const AdmittedMoEPipelineMemory &admission)
    {
        std::ostringstream out;
        out << "pipeline-expert-authority-v1/" << admission.stages().size() << '/';
        for (const auto &stage : admission.stages()) appendPipelineStage(out, stage.topology());
        return out.str();
    }
}
