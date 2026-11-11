/**
 * @file PreparedWeightAuthorityIdentity.cpp
 * @brief Structural identity shared by initial admission and prepared-context reuse.
 *
 * Stage and model-wide consumers use the same ordered policy serialization.
 * Only metadata is inspected; model and cache payloads are never hashed. Active
 * request controls may change within the unchanged admitted capacity envelope.
 * Complete runtime handoff borrows the original immutable admission after exact plan
 * validation; participant factories retain their existing geometry checks.
 *
 * Local and global pipeline stages share the original rank-owned prepared
 * source and PMA handoff. Global topology remains immutable metadata; remote
 * execution, activation transport and model-context reuse are not impersonated.
 */
#include "PreparedWeightAuthorityIdentity.h"
#include "execution/config/RuntimeConfig.h"
#include "backends/NativeContextPreparation.h"
#include "loaders/ResolvedImmutableRowTablePlacement.h"
#include "execution/local_execution/graph/RetainedHostServiceDeclarations.h"
#include "execution/moe/MoEOverlayHostServiceContract.h"
#include "execution/moe/MoEExpertOverlayExecutionPlan.h"
#include <algorithm>
#include <stdexcept>
#include "planning/MoEPipelineMemoryAdmission.h"
#include "execution/local_execution/engine/PrefillBucketUtils.h"
#include "utils/DebugEnv.h"
#include <sstream>
#include <iomanip>

namespace llaminar2
{
    std::vector<DeviceId> admittedOverlayPreparedWeightDevices(
        const MoERoutedExpertPlacementPlan& plan, const MoEExpertOverlayExecutionPlan& execution)
    {
        // Initial import may precede placement admission. Without original
        // layer ownership there is no proof permitting any endpoint omission.
        if (plan.placements.empty()) return execution.currentRankPlan().local_devices;
        // This projection needs only original membership, never whole-expert owner
        // rows. A joint fragment must not accidentally mint a whole owner here.
        const auto participants = MoEExpertOwnerMap::declaredParticipants(plan, {.reject_tensor_sharded_domains = false});
        const auto admitted = MoEOverlayHostServiceContract::admittedExpertParticipants(plan, participants);
        const auto& rank = execution.currentRankPlan();
        const auto endpoint = [&](const auto& participant, DeviceId device) {
            return participant.world_rank_known && participant.world_rank == rank.world_rank && participant.device == device;
        };
        const auto denseOwner = [&](DeviceId device) {
            for (const auto& domain : execution.domains) {
                const bool dense = domain.name == plan.continuation_domain ||
                    std::find(rank.root_weight_domains.begin(), rank.root_weight_domains.end(), domain.name) != rank.root_weight_domains.end() ||
                    std::find(rank.shared_expert_weight_domains.begin(), rank.shared_expert_weight_domains.end(), domain.name) != rank.shared_expert_weight_domains.end();
                if (!dense) continue;
                for (const auto& participant : domain.participants)
                    if (participant.world_rank_known && participant.world_rank == rank.world_rank && participant.local_device == device)
                        return true;
            }
            return false;
        };
        std::vector<DeviceId> result;
        for (const auto device : rank.local_devices) {
            if (!device.is_valid()) throw std::logic_error("Prepared-weight authority has an invalid original endpoint");
            const bool declared = std::any_of(participants.begin(), participants.end(),
                [&](const auto& participant) { return endpoint(participant, device); });
            const bool live = std::any_of(admitted.begin(), admitted.end(),
                [&](const auto& participant) { return endpoint(participant, device); });
            // Unclassified original devices are retained conservatively. Only
            // a routed membership explicitly removed by the shared admitted
            // projection may disappear from the prepared-record authority set.
            if ((!declared || live || denseOwner(device)) &&
                std::find(result.begin(), result.end(), device) == result.end()) result.push_back(device);
        }
        return result;
    }

    bool retainedMTPWeightAuthorityMatches(const MTPRuntimeConfig &retained, const MTPRuntimeConfig &requested)
    {
        return resolveMTPRetainedDraftCapacity(retained) == resolveMTPRetainedDraftCapacity(requested) &&
            retained.max_request_batch == requested.max_request_batch &&
            retained.sidecar_dense_policy == requested.sidecar_dense_policy &&
            retained.terminal_head_policy == requested.terminal_head_policy;
    }

    void restoreRetainedAdmittedRuntime(RuntimeConfig& current,const RuntimeConfig& original,int world_rank)
    {
        if(world_rank<0)throw std::invalid_argument("Retained runtime admission requires its original physical rank");
        const auto& source=original.native_context_preparation;
        const auto& requested_source=current.native_context_preparation;
        if(requested_source && (!source || requested_source.get()!=source.get() ||
            requested_source.owner_before(source) || source.owner_before(requested_source)))
            throw std::logic_error("Retained runtime admission cannot replace its original native source");
        if(source && source->memory()->worldRank()!=world_rank)
            throw std::logic_error("Retained runtime source crossed its original physical rank");
        if(original.attention_participants && current.attention_setup_request!=original.attention_setup_request)
            throw std::invalid_argument("Retained attention admission has a different startup policy request");
        const auto& placement=original.ngram_table_placement;
        if(placement){
            if(placement->worldRank()!=world_rank)
                throw std::logic_error("Retained lookup placement crossed its original physical rank");
            placement->requireRequestedPolicy(current.ngram_table_policy);
        }
        if(current.ngram_table_placement && (!placement || *current.ngram_table_placement!=*placement))
            throw std::logic_error("Retained runtime admission conflicts with an existing lookup placement");
        if(original.retained_host_services){
            if(original.retained_host_services->worldRank()!=world_rank)
                throw std::logic_error("Retained CPU service declaration crossed its original physical rank");
            // This is the same final placement selector used by the factory.
            // In particular, a CPU I/O role cannot be restored without its
            // original selected source. No instance is bound or materialized.
            (void)original.retained_host_services->selectedForLookup(placement.get());
        }
        // Stage the only allocating metadata copy before publishing any field.
        auto cpu_storage=original.cpu_serial_expert_storage;
        current.native_context_preparation=source;
        current.resident_graph_rows=original.resident_graph_rows;
        current.moe_routed_prefill.overlay_segment_rows=original.moe_routed_prefill.overlay_segment_rows;
        current.ngram_table_placement=placement;
        current.attention_participants=original.attention_participants;
        current.cpu_serial_expert_storage=std::move(cpu_storage);
        current.retained_host_services=original.retained_host_services;
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
        if (config.ngram_table_policy.isFixedHeadSharded())
            out << "prefill_cache_limit_explicit="
                << debugEnv().execution.prefill_graph_max_cached_buckets_explicit << ',';
        append_optional_bytes(config.max_gpu_memory_mb);
        append_optional_bytes(config.max_cpu_memory_mb);
        append_string(config.activation_precision);
        append_string(config.kv_cache_precision);
        out << ";ngram=" << static_cast<int>(config.ngram_table_policy.storage) << ','
            << config.ngram_table_policy.window_bytes << ',' << config.ngram_table_policy.window_count
            << ",distribution=" << static_cast<int>(config.ngram_table_policy.distribution);
        if(config.ngram_table_policy.isHostAdaptive())
        {
            const auto &a=config.ngram_table_policy.host_adaptive;
            out << ",adaptive=" << a.maximum_cache_rows << ',' << a.cache_physical_byte_limit << ','
                << a.policy.aging_accesses << ',' << a.policy.protected_numerator << ',' << a.policy.protected_denominator;
        }
        if(config.ngram_table_policy.isGPUAdaptive())
        {
            const auto &g=config.ngram_table_policy.gpu_adaptive;
            out << ",gpu_adaptive=" << g.maximum_gpu_cache_rows << ',' << g.gpu_cache_physical_byte_limit << ','
                << g.gpu_policy.aging_accesses << ',' << g.gpu_policy.protected_numerator << ',' << g.gpu_policy.protected_denominator
                << ",subordinate=" << g.host.maximum_cache_rows << ',' << g.host.cache_physical_byte_limit << ','
                << g.host.policy.aging_accesses << ',' << g.host.policy.protected_numerator << ',' << g.host.policy.protected_denominator;
            g.placement.validate();
            if(g.placement.mode==AdaptiveGPUCachePlacementMode::DeclaredRankLocal)
            {
                // Retained authorities cannot normalize away authored physical
                // coordinate order, C0 owners, individual limits or coordinator.
                out << ",gpu_placement=" << static_cast<int>(g.placement.mode) << ",owners=" << g.placement.owners.size();
                for(const auto& owner:g.placement.owners)
                    out << ';' << owner.participant.world_rank << ',' << static_cast<int>(owner.participant.backend) << ','
                        << owner.participant.ordinal << ',' << owner.maximum_cache_rows << ',' << owner.cache_physical_byte_limit;
                const auto& coordinator=*g.placement.coordinator;
                out << ",coordinator=" << coordinator.world_rank << ',' << static_cast<int>(coordinator.backend) << ',' << coordinator.ordinal;
            }
        }
        // Explicit HC-up correction changes stage workspace capacity even
        // though every prepared source weight byte remains identical. A
        // retained capacity authority must reject that setup-policy change.
        if (debugEnv().gemm.qfn_hc_up_activation != QuantizedActivationOperandPrecision::SingleQ8)
            out << ";qfn_hc_up_activation="
                << quantizedActivationOperandPrecisionName(debugEnv().gemm.qfn_hc_up_activation);
        if (debugEnv().gemm.qfn_gdn_output_activation != QuantizedActivationOperandPrecision::SingleQ8)
            out << ";qfn_gdn_output_activation="
                << quantizedActivationOperandPrecisionName(debugEnv().gemm.qfn_gdn_output_activation);
        if (debugEnv().gemm.qfn_fa_output_activation != QuantizedActivationOperandPrecision::SingleQ8)
            out << ";qfn_fa_output_activation="
                << quantizedActivationOperandPrecisionName(debugEnv().gemm.qfn_fa_output_activation);
        if (debugEnv().gemm.qfn_gdn_qkv_activation != QuantizedActivationOperandPrecision::SingleQ8)
            out << ";qfn_gdn_qkv_activation="
                << quantizedActivationOperandPrecisionName(debugEnv().gemm.qfn_gdn_qkv_activation);
        return out.str();
    }

    namespace
    {
        /** @brief Encode the original global stage declaration without mutable runtime observations. */
        void appendGlobalPipeline(std::ostream &out, const std::optional<GlobalPPTopology> &global)
        {
            if (!global) return; // Preserve existing local prepared identities.
            out << "global-stage-ownership-v1/" << global->world_size << '/' << global->total_layers << '/';
            for (const auto &stage : global->stages)
            {
                out << stage.stage_id << ',' << stage.first_layer << ',' << stage.last_layer << ','
                    << stage.has_embedding << ',' << stage.has_lm_head << ',' << stage.owning_rank << ','
                    << stage.is_global_tp << ',' << static_cast<int>(stage.inner_mode) << ','
                    << static_cast<int>(stage.backend) << '/' << stage.domain_name.size() << ':' << stage.domain_name;
                for (const auto &device : stage.devices)
                    out << '/' << device.hostname.size() << ':' << device.hostname << ','
                        << static_cast<int>(device.device_type) << ',' << device.device_ordinal << ',' << device.numa_node;
                out << "/weights";
                for (const auto weight : stage.tp_weights) out << ',' << std::hexfloat << weight << std::defaultfloat;
                out << ';';
            }
        }

        /** @brief Append one global interval and its complete capacity-affecting policy. */
        void appendPipelineStage(std::ostream &out, const ResolvedMoEPipelineStage &stage)
        {
            const auto &scope = stage.scope();
            const auto &config = stage.config();
            const auto identity = routedWeightAuthorityRequestIdentity(config.moe_routed_expert_plan, config);
            const auto &cache = config.prefix_cache;
            out << scope.first_layer << ',' << scope.last_layer << ',' << scope.has_embedding << ',' << scope.has_lm_head
                << ',' << static_cast<int>(scope.generation_domain)
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
        appendGlobalPipeline(out, topology.globalPipeline());
        for (const auto &stage : topology.pipelineStages()) appendPipelineStage(out, stage);
        return out.str();
    }

    std::string pipelineRoutedWeightAuthorityRequestIdentity(const AdmittedMoEPipelineMemory &admission)
    {
        std::ostringstream out;
        out << "pipeline-expert-authority-v1/" << admission.stages().size() << '/';
        appendGlobalPipeline(out, admission.globalPipeline());
        for (const auto &stage : admission.stages()) appendPipelineStage(out, stage.topology());
        return out.str();
    }
}
