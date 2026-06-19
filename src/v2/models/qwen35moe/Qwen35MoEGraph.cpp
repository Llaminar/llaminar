/**
 * @file Qwen35MoEGraph.cpp
 * @brief Qwen 3.5 MoE compute graph builder implementation
 */

#include "Qwen35MoEGraph.h"
#include "Qwen35MoESchema.h"
#include "../../utils/Logger.h"
#include "../../execution/compute_stages/ComputeStageFactory.h"
#include "../../execution/compute_stages/stages/MoEExpertDispatchStage.h"
#include "../../execution/compute_stages/stages/MoERoutingStage.h"
#include "../../execution/compute_stages/stages/MoEExpertComputeStage.h"
#include "../../execution/compute_stages/stages/MoELocalExpertStage.h"
#include "../../execution/compute_stages/stages/MoESparseDispatchStage.h"
#include "../../execution/compute_stages/stages/MoESparseReturnReduceStage.h"
#include "../../execution/moe/MoEExpertParallelPlan.h"
#include "../../execution/moe/MoEExpertOwnerMap.h"
#include "../../execution/moe/MoEExpertOverlayRuntimePlan.h"
#include "../../execution/moe/MoEOverlaySparseCollective.h"
#include "../../execution/prefix_cache/PrefixCacheFingerprint.h"
#include "../../memory/BufferId.h"
#include "../../execution/local_execution/graph/GraphResolver.h"
#include "../../tensors/Tensors.h"
#include "../../utils/DebugEnv.h"
#include "../../utils/PerfStatsCollector.h"

#include <algorithm>
#include <cctype>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace llaminar2
{

    namespace
    {
        const ExpertLayerPlacement *findExpertOverlayPlacement(
            const MoEExpertParallelPlan &plan,
            int layer_idx)
        {
            auto it = std::find_if(plan.placements.begin(), plan.placements.end(),
                                   [layer_idx](const ExpertLayerPlacement &placement)
                                   {
                                       return placement.layer == layer_idx;
                                   });
            return it == plan.placements.end() ? nullptr : &(*it);
        }

        bool isUsableExpertOverlayPlacement(
            const ExpertLayerPlacement &placement,
            const MoEExpertParallelPlan &plan,
            int num_experts,
            int layer_idx)
        {
            if (static_cast<int>(placement.routed_expert_tier.size()) != num_experts)
            {
                LOG_ERROR("[Qwen35MoEGraph] Expert overlay placement for layer " << layer_idx
                                                                                 << " covers " << placement.routed_expert_tier.size()
                                                                                 << " experts, expected " << num_experts);
                return false;
            }

            for (int expert = 0; expert < num_experts; ++expert)
            {
                const int tier_index = placement.routed_expert_tier[static_cast<size_t>(expert)];
                if (tier_index < 0 || tier_index >= static_cast<int>(plan.routed_tiers.size()))
                {
                    LOG_ERROR("[Qwen35MoEGraph] Expert overlay placement for layer " << layer_idx
                                                                                     << " maps expert " << expert << " to invalid tier " << tier_index);
                    return false;
                }
            }

            return true;
        }

        std::vector<bool> expertMaskForTier(
            const ExpertLayerPlacement &placement,
            int num_experts,
            int tier_index)
        {
            std::vector<bool> mask(static_cast<size_t>(num_experts), false);
            for (int expert = 0; expert < num_experts; ++expert)
            {
                mask[static_cast<size_t>(expert)] =
                    placement.routed_expert_tier[static_cast<size_t>(expert)] == tier_index;
            }
            return mask;
        }

        std::string nodeSuffixForTier(const ExpertRoutedTier &tier, int tier_index)
        {
            std::string suffix = "tier" + std::to_string(tier_index);
            if (!tier.name.empty())
            {
                suffix += "_";
                for (unsigned char ch : tier.name)
                    suffix += std::isalnum(ch) ? static_cast<char>(std::tolower(ch)) : '_';
            }
            return suffix;
        }

        std::shared_ptr<MoEExpertOverlayRuntimePlan> runtimePlanForGraph(
            const GraphConfig &config)
        {
            if (config.moe.expert_overlay_runtime_plan)
                return config.moe.expert_overlay_runtime_plan;
            if (!config.moe.expert_parallel_plan)
                return nullptr;
            return resolveMoEExpertOverlayRuntimePlan(config.moe.expert_parallel_plan);
        }

        int continuationRootParticipant(const MoEExpertParallelPlan &plan)
        {
            return std::max(0, plan.continuation_domain_spec.logical_root_participant);
        }

        int participantCountForGraphNativeOverlay(
            const MoEExpertOwnerMap &owner_map,
            int continuation_root_participant)
        {
            int max_participant = std::max(0, continuation_root_participant);
            for (const auto &participant : owner_map.participants())
                max_participant = std::max(max_participant, participant.participant_id);
            return max_participant + 1;
        }

        std::vector<int> participantsWithLast(int participant_count, int last_participant)
        {
            std::vector<int> participants;
            participants.reserve(static_cast<size_t>(std::max(0, participant_count)));
            for (int participant = 0; participant < participant_count; ++participant)
            {
                if (participant != last_participant)
                    participants.push_back(participant);
            }
            if (last_participant >= 0 && last_participant < participant_count)
                participants.push_back(last_participant);
            return participants;
        }

        bool allExpertsEnabled(const std::vector<bool> &expert_mask, int num_experts)
        {
            return expert_mask.empty() ||
                   (expert_mask.size() == static_cast<size_t>(num_experts) &&
                    std::all_of(expert_mask.begin(), expert_mask.end(),
                                [](bool enabled)
                                { return enabled; }));
        }

        bool runtimeTableHasUsableDecodeBank(
            IMoERuntimeTable *runtime_table,
            int layer_idx,
            int num_experts,
            int top_k,
            bool require_full_local_descriptors)
        {
            if (!runtime_table || layer_idx < 0)
                return false;

            const auto &state = runtime_table->hostLayerState(layer_idx);
            if (state.active_bank > 1 ||
                state.active_epoch == 0 ||
                state.expert_count != static_cast<uint32_t>(num_experts) ||
                state.top_k != static_cast<uint32_t>(top_k))
            {
                return false;
            }

            const auto &bank = state.banks[state.active_bank];
            if (bank.epoch != state.active_epoch ||
                bank.expert_count != static_cast<uint32_t>(num_experts))
            {
                return false;
            }

            if (!require_full_local_descriptors)
                return true;

            for (int expert = 0; expert < num_experts; ++expert)
            {
                const auto &desc = bank.experts[static_cast<size_t>(expert)];
                if (bank.local_compute_mask[static_cast<size_t>(expert)] != 1u ||
                    desc.logical_expert_id != expert ||
                    desc.local_slot < 0 ||
                    !desc.gate.valid() ||
                    !desc.up.valid() ||
                    !desc.down.valid() ||
                    !hasMoEExpertFlag(desc.flags, DeviceMoEExpertFlags::Valid) ||
                    !hasMoEExpertFlag(desc.flags, DeviceMoEExpertFlags::Resident) ||
                    !hasMoEExpertFlag(desc.flags, DeviceMoEExpertFlags::LocalCompute))
                {
                    return false;
                }
            }
            return true;
        }

        bool initializeFullLocalDecodeRuntimeTable(
            IMoERuntimeTable *runtime_table,
            int layer_idx,
            int num_experts,
            int top_k,
            int d_model,
            int expert_intermediate,
            const std::vector<ITensorGemm *> &gate_gemms,
            const std::vector<ITensorGemm *> &up_gemms,
            const std::vector<ITensorGemm *> &down_gemms,
            void *stream,
            const std::string &context)
        {
            if (!runtime_table || layer_idx < 0)
                return false;

            if (runtimeTableHasUsableDecodeBank(
                    runtime_table, layer_idx, num_experts, top_k,
                    /*require_full_local_descriptors=*/true))
            {
                return true;
            }

            const auto &state = runtime_table->hostLayerState(layer_idx);
            if (state.active_epoch != 0)
            {
                LOG_ERROR("[Qwen35MoEGraph] " << context
                                              << ": refusing to replace active MoE decode runtime bank for layer "
                                              << layer_idx << " epoch " << state.active_epoch);
                return false;
            }

            if (gate_gemms.size() != static_cast<size_t>(num_experts) ||
                up_gemms.size() != static_cast<size_t>(num_experts) ||
                down_gemms.size() != static_cast<size_t>(num_experts))
            {
                LOG_ERROR("[Qwen35MoEGraph] " << context
                                              << ": prepared expert GEMM vector sizes do not match num_experts="
                                              << num_experts);
                return false;
            }

            MoEPlacementUpdate update;
            update.epoch = 1;
            update.expert_count = static_cast<uint32_t>(num_experts);
            update.experts.resize(static_cast<size_t>(num_experts));
            update.local_compute_mask.assign(static_cast<size_t>(num_experts), 1u);
            update.replica_role.assign(static_cast<size_t>(num_experts),
                                       static_cast<uint8_t>(DeviceMoEReplicaRole::Primary));

            const uint32_t flags = toMoEExpertFlags(DeviceMoEExpertFlags::Valid |
                                                    DeviceMoEExpertFlags::Resident |
                                                    DeviceMoEExpertFlags::PreferredOwner |
                                                    DeviceMoEExpertFlags::LocalCompute);

            for (int expert = 0; expert < num_experts; ++expert)
            {
                auto *gate = gate_gemms[static_cast<size_t>(expert)];
                auto *up = up_gemms[static_cast<size_t>(expert)];
                auto *down = down_gemms[static_cast<size_t>(expert)];
                if (!gate || !up || !down)
                {
                    LOG_ERROR("[Qwen35MoEGraph] " << context
                                                  << ": missing prepared GEMM engine for expert "
                                                  << expert << " layer " << layer_idx);
                    return false;
                }

                DeviceMoEExpertDescriptor desc;
                if (!gate->exportNativeVNNIMatrixDesc(desc.gate) ||
                    !up->exportNativeVNNIMatrixDesc(desc.up) ||
                    !down->exportNativeVNNIMatrixDesc(desc.down))
                {
                    LOG_ERROR("[Qwen35MoEGraph] " << context
                                                  << ": prepared GEMM engines for expert "
                                                  << expert << " layer " << layer_idx
                                                  << " cannot export native-VNNI descriptors");
                    return false;
                }

                if (desc.gate.n != expert_intermediate || desc.gate.k != d_model ||
                    desc.up.n != expert_intermediate || desc.up.k != d_model ||
                    desc.down.n != d_model || desc.down.k != expert_intermediate)
                {
                    LOG_ERROR("[Qwen35MoEGraph] " << context
                                                  << ": native-VNNI descriptor shape mismatch for expert "
                                                  << expert << " layer " << layer_idx);
                    return false;
                }

                desc.logical_expert_id = expert;
                desc.owner_participant = 0;
                desc.local_slot = expert;
                desc.flags = flags;
                update.experts[static_cast<size_t>(expert)] = desc;
            }

            try
            {
                runtime_table->prepareInactiveBank(layer_idx, update);
                runtime_table->flipActiveBank(layer_idx, update.epoch, stream);
            }
            catch (const std::exception &ex)
            {
                LOG_ERROR("[Qwen35MoEGraph] " << context
                                              << ": failed to publish MoE decode runtime bank for layer "
                                              << layer_idx << ": " << ex.what());
                return false;
            }

            return runtimeTableHasUsableDecodeBank(
                runtime_table, layer_idx, num_experts, top_k,
                /*require_full_local_descriptors=*/true);
        }

        MoEOverlayCollectiveKey graphNativeMoEKey(
            int layer_idx,
            int tier_idx,
            int target_participant,
            MoEOverlayCollectiveDirection direction,
            bool mtp_sidecar_context,
            int mtp_depth_idx)
        {
            if (mtp_sidecar_context)
            {
                return makeMTPMoEOverlayCollectiveKey(
                    1,
                    0,
                    std::max(0, mtp_depth_idx),
                    layer_idx,
                    tier_idx,
                    std::max(0, target_participant),
                    std::max(0, target_participant),
                    direction);
            }

            return makeMoEOverlayCollectiveKey(
                1,
                0,
                layer_idx,
                tier_idx,
                std::max(0, target_participant),
                std::max(0, target_participant),
                direction);
        }

        std::string boolField(bool value)
        {
            return value ? "true" : "false";
        }

        void appendAddressVectorFields(
            std::vector<PrefixFingerprintField> &fields,
            const std::string &prefix,
            const std::vector<GlobalDeviceAddress> &addresses)
        {
            fields.push_back({prefix + ".count", std::to_string(addresses.size())});
            for (size_t index = 0; index < addresses.size(); ++index)
                fields.push_back({prefix + "." + std::to_string(index), addresses[index].toString()});
        }

        void appendRankVectorFields(
            std::vector<PrefixFingerprintField> &fields,
            const std::string &prefix,
            const std::vector<int> &ranks)
        {
            fields.push_back({prefix + ".count", std::to_string(ranks.size())});
            for (size_t index = 0; index < ranks.size(); ++index)
                fields.push_back({prefix + "." + std::to_string(index), std::to_string(ranks[index])});
        }

        void appendWeightVectorFields(
            std::vector<PrefixFingerprintField> &fields,
            const std::string &prefix,
            const std::vector<float> &weights)
        {
            fields.push_back({prefix + ".count", std::to_string(weights.size())});
            for (size_t index = 0; index < weights.size(); ++index)
                fields.push_back({prefix + "." + std::to_string(index), std::to_string(weights[index])});
        }

        void appendDenseDomainFingerprintFields(
            std::vector<PrefixFingerprintField> &fields,
            const ExecutionDomainDefinition &domain,
            const std::string &prefix)
        {
            fields.push_back({prefix + ".name", domain.name});
            fields.push_back({prefix + ".scope", executionDomainScopeToString(domain.scope)});
            fields.push_back({prefix + ".backend", collectiveBackendTypeToString(domain.backend)});
            fields.push_back({prefix + ".compute_kind", executionDomainComputeKindToString(domain.compute_kind)});
            fields.push_back({prefix + ".owner_rank", domain.owner_rank ? std::to_string(*domain.owner_rank) : "-1"});
            appendAddressVectorFields(fields, prefix + ".participant", domain.participants);
            appendRankVectorFields(fields, prefix + ".rank", domain.ranks);
            appendWeightVectorFields(fields, prefix + ".weight", domain.weights);
        }

        void appendExpertDomainFingerprintFields(
            std::vector<PrefixFingerprintField> &fields,
            const ExpertComputeDomain &domain,
            const std::string &prefix)
        {
            fields.push_back({prefix + ".name", domain.name});
            fields.push_back({prefix + ".kind", toString(domain.kind)});
            fields.push_back({prefix + ".backend", collectiveBackendTypeToString(domain.backend)});
            fields.push_back({prefix + ".compute_kind", toString(domain.compute_kind)});
            fields.push_back({prefix + ".owner_rank", std::to_string(domain.owner_rank)});
            appendAddressVectorFields(fields, prefix + ".participant", domain.participants);
            appendRankVectorFields(fields, prefix + ".rank", domain.world_ranks);
            appendWeightVectorFields(fields, prefix + ".weight", domain.weights);
        }

        void appendExpertOverlayPlanFingerprintFields(
            std::vector<PrefixFingerprintField> &fields,
            const MoEExpertParallelPlan &plan,
            const std::string &scope)
        {
            fields.push_back({scope + ".enabled", boolField(plan.enabled)});
            fields.push_back({scope + ".execution_kind", toString(plan.execution_kind)});
            fields.push_back({scope + ".continuation_domain", plan.continuation_domain});
            fields.push_back({scope + ".base_model_domain", plan.base_model_domain});
            fields.push_back({scope + ".effective_base_model_domain", plan.effectiveBaseModelDomain()});
            fields.push_back({scope + ".shared_expert_domain", plan.shared_expert_domain});
            fields.push_back({scope + ".residency_policy", toString(plan.residency_policy)});
            fields.push_back({scope + ".continuation_spec.domain", plan.continuation_domain_spec.domain});
            fields.push_back({scope + ".continuation_spec.logical_root_participant",
                              std::to_string(plan.continuation_domain_spec.logical_root_participant)});
            fields.push_back({scope + ".continuation_spec.dense_tp_enabled",
                              boolField(plan.continuation_domain_spec.dense_tp_enabled)});
            fields.push_back({scope + ".continuation_spec.hidden_layout",
                              toString(plan.continuation_domain_spec.hidden_layout)});
            fields.push_back({scope + ".continuation_spec.shared_expert_uses_dense_tp",
                              boolField(plan.continuation_domain_spec.shared_expert_uses_dense_tp)});

            fields.push_back({scope + ".dense_domain.count", std::to_string(plan.dense_domains.size())});
            for (size_t index = 0; index < plan.dense_domains.size(); ++index)
                appendDenseDomainFingerprintFields(
                    fields,
                    plan.dense_domains[index],
                    scope + ".dense_domain." + std::to_string(index));

            fields.push_back({scope + ".expert_domain.count", std::to_string(plan.domains.size())});
            for (size_t index = 0; index < plan.domains.size(); ++index)
                appendExpertDomainFingerprintFields(
                    fields,
                    plan.domains[index],
                    scope + ".expert_domain." + std::to_string(index));

            fields.push_back({scope + ".routed_tier.count", std::to_string(plan.routed_tiers.size())});
            for (size_t index = 0; index < plan.routed_tiers.size(); ++index)
            {
                const auto &tier = plan.routed_tiers[index];
                const std::string prefix = scope + ".routed_tier." + std::to_string(index);
                fields.push_back({prefix + ".name", tier.name});
                fields.push_back({prefix + ".domain", tier.domain});
                fields.push_back({prefix + ".priority", std::to_string(tier.priority)});
                fields.push_back({prefix + ".max_experts_per_layer", std::to_string(tier.max_experts_per_layer)});
                fields.push_back({prefix + ".memory_budget_bytes", std::to_string(tier.memory_budget_bytes)});
                fields.push_back({prefix + ".fallback", boolField(tier.fallback)});
            }

            fields.push_back({scope + ".placement.count", std::to_string(plan.placements.size())});
            for (size_t placement_index = 0; placement_index < plan.placements.size(); ++placement_index)
            {
                const auto &placement = plan.placements[placement_index];
                const std::string prefix = scope + ".placement." + std::to_string(placement_index);
                fields.push_back({prefix + ".layer", std::to_string(placement.layer)});
                fields.push_back({prefix + ".routed_expert_tier.count",
                                  std::to_string(placement.routed_expert_tier.size())});
                for (size_t expert = 0; expert < placement.routed_expert_tier.size(); ++expert)
                    fields.push_back({prefix + ".routed_expert_tier." + std::to_string(expert),
                                      std::to_string(placement.routed_expert_tier[expert])});
            }
        }

        void appendExpertOverlayRuntimeFingerprintFields(
            std::vector<PrefixFingerprintField> &fields,
            const MoEExpertOverlayRuntimePlan &runtime_plan,
            const std::string &scope)
        {
            fields.push_back({scope + ".enabled", "true"});
            fields.push_back({scope + ".current_world_rank", std::to_string(runtime_plan.currentWorldRank())});

            fields.push_back({scope + ".domain.count", std::to_string(runtime_plan.domains().size())});
            for (size_t domain_index = 0; domain_index < runtime_plan.domains().size(); ++domain_index)
            {
                const auto &domain = runtime_plan.domains()[domain_index];
                const std::string prefix = scope + ".domain." + std::to_string(domain_index);
                fields.push_back({prefix + ".name", domain.name});
                fields.push_back({prefix + ".kind", toString(domain.kind)});
                fields.push_back({prefix + ".backend", collectiveBackendTypeToString(domain.backend)});
                fields.push_back({prefix + ".compute_kind", toString(domain.compute_kind)});
                fields.push_back({prefix + ".primary_participant", domain.primary_participant.toString()});
                fields.push_back({prefix + ".primary_device", domain.primary_device.to_string()});
                fields.push_back({prefix + ".primary_world_rank", std::to_string(domain.primary_world_rank)});
                fields.push_back({prefix + ".primary_world_rank_known", boolField(domain.primary_world_rank_known)});
                fields.push_back({prefix + ".owner_rank", std::to_string(domain.owner_rank)});
                fields.push_back({prefix + ".primary_is_local", boolField(domain.primary_is_local)});
                fields.push_back({prefix + ".primary_owned_by_current_rank",
                                  boolField(domain.primary_owned_by_current_rank)});
                fields.push_back({prefix + ".local_reachable_for_mvp", boolField(domain.local_reachable_for_mvp)});
                fields.push_back({prefix + ".requires_domain_scoped_collective_context",
                                  boolField(domain.requires_domain_scoped_collective_context)});
                fields.push_back({prefix + ".domain_scoped_collective_context_ready",
                                  boolField(domain.domain_scoped_collective_context_ready)});
                fields.push_back({prefix + ".multi_participant_execution_pending",
                                  boolField(domain.multi_participant_execution_pending)});
                fields.push_back({prefix + ".pending_reason", domain.pending_reason});

                fields.push_back({prefix + ".participant.count", std::to_string(domain.participants.size())});
                for (size_t participant_index = 0; participant_index < domain.participants.size(); ++participant_index)
                {
                    const auto &participant = domain.participants[participant_index];
                    const std::string participant_prefix =
                        prefix + ".participant." + std::to_string(participant_index);
                    fields.push_back({participant_prefix + ".address", participant.address.toString()});
                    fields.push_back({participant_prefix + ".participant_index",
                                      std::to_string(participant.participant_index)});
                    fields.push_back({participant_prefix + ".world_rank", std::to_string(participant.world_rank)});
                    fields.push_back({participant_prefix + ".world_rank_known",
                                      boolField(participant.world_rank_known)});
                    fields.push_back({participant_prefix + ".owned_by_current_rank",
                                      boolField(participant.owned_by_current_rank)});
                    fields.push_back({participant_prefix + ".locally_addressable",
                                      boolField(participant.locally_addressable)});
                    fields.push_back({participant_prefix + ".local_device", participant.local_device.to_string()});
                }
            }

            fields.push_back({scope + ".routed_tier.count", std::to_string(runtime_plan.routedTiers().size())});
            for (size_t tier_index = 0; tier_index < runtime_plan.routedTiers().size(); ++tier_index)
            {
                const auto &tier = runtime_plan.routedTiers()[tier_index];
                const std::string prefix = scope + ".routed_tier." + std::to_string(tier_index);
                fields.push_back({prefix + ".tier_index", std::to_string(tier.tier_index)});
                fields.push_back({prefix + ".name", tier.tier.name});
                fields.push_back({prefix + ".domain_name", tier.domain_name});
                fields.push_back({prefix + ".primary_device", tier.primary_device.to_string()});
                fields.push_back({prefix + ".local_reachable_for_mvp", boolField(tier.local_reachable_for_mvp)});
                fields.push_back({prefix + ".multi_participant_execution_pending",
                                  boolField(tier.multi_participant_execution_pending)});
            }
        }

        DeviceId participantDeviceForGraphNativeOverlay(
            const MoEExpertOwnerMap &owner_map,
            int participant_id)
        {
            const auto *participant = owner_map.participantForId(participant_id);
            if (!participant || !participant->device.is_valid())
                return DeviceId::cpu();
            return participant->device;
        }

        const ExpertComputeDomain *expertDomainForTier(
            const MoEExpertParallelPlan &plan,
            const ExpertRoutedTier &tier)
        {
            auto it = std::find_if(plan.domains.begin(), plan.domains.end(),
                                   [&](const ExpertComputeDomain &domain)
                                   {
                                       return domain.name == tier.domain;
                                   });
            return it == plan.domains.end() ? nullptr : &(*it);
        }

        bool isLocalTPReplicatedExpertsTier(
            const MoEExpertParallelPlan &plan,
            const ExpertRoutedTier &tier)
        {
            const auto *domain = expertDomainForTier(plan, tier);
            return domain &&
                   domain->kind == ExpertDomainKind::LocalTP &&
                   domain->compute_kind == ExpertDomainComputeKind::ReplicatedExperts &&
                   domain->participants.size() > 1;
        }

        int participantIdForTierDevice(
            const MoEExpertOwnerMap &owner_map,
            int tier_index,
            DeviceId device)
        {
            for (const auto &participant : owner_map.participants())
            {
                if (participant.tier_idx == tier_index &&
                    participant.device.is_valid() &&
                    participant.device == device)
                {
                    return participant.participant_id;
                }
            }
            return -1;
        }
    } // namespace

    // =========================================================================
    // Constructors
    // =========================================================================

    Qwen35MoEGraph::Qwen35MoEGraph(
        std::shared_ptr<ModelContext> model_ctx,
        std::shared_ptr<IMPIContext> mpi_ctx,
        const GraphConfig &config)
        : Qwen35Graph(std::move(model_ctx), std::move(mpi_ctx), config)
    {
    }

    Qwen35MoEGraph::Qwen35MoEGraph(
        const GraphConfig &config,
        std::shared_ptr<IMPIContext> mpi_ctx)
        : Qwen35Graph(config, std::move(mpi_ctx))
    {
    }

    Qwen35MoEGraph::ScopedMTPGraphContext::ScopedMTPGraphContext(
        Qwen35MoEGraph &graph,
        int depth_idx)
        : graph(graph),
          previous_active(graph.mtp_graph_context_active_),
          previous_depth_idx(graph.mtp_graph_depth_idx_)
    {
        graph.mtp_graph_context_active_ = true;
        graph.mtp_graph_depth_idx_ = depth_idx;
    }

    Qwen35MoEGraph::ScopedMTPGraphContext::~ScopedMTPGraphContext()
    {
        graph.mtp_graph_context_active_ = previous_active;
        graph.mtp_graph_depth_idx_ = previous_depth_idx;
    }

    ComputeGraph Qwen35MoEGraph::buildMTPGraph(
        int depth_idx,
        const MTPDepthWeights &weights,
        const MTPForwardInput &input,
        MTPForwardOutput &output)
    {
        ScopedMTPGraphContext context(*this, depth_idx);
        return Qwen35Graph::buildMTPGraph(depth_idx, weights, input, output);
    }

    ComputeGraph Qwen35MoEGraph::buildMTPGraph(
        int depth_idx,
        const MTPDepthWeightBindings &bindings,
        const MTPForwardInput &input,
        MTPForwardOutput &output)
    {
        ScopedMTPGraphContext context(*this, depth_idx);
        return Qwen35Graph::buildMTPGraph(depth_idx, bindings, input, output);
    }

    void Qwen35MoEGraph::resetState()
    {
        Qwen35Graph::resetState();

        if (config_.moe.decode_histogram)
            config_.moe.decode_histogram->resetWindow();

        for (auto &[key, table] : moe_runtime_tables_)
        {
            (void)key;
            if (table)
                table->resetDecodeHistogramCounts();
        }
    }

    void Qwen35MoEGraph::appendPrefixCacheFingerprintMaterial(PrefixFingerprintMaterial &material) const
    {
        material.moe.push_back({"graph.num_experts", std::to_string(config_.moe.num_experts)});
        material.moe.push_back({"graph.top_k", std::to_string(config_.moe.top_k)});

        if (config_.moe.expert_parallel_plan)
        {
            appendExpertOverlayPlanFingerprintFields(
                material.moe,
                *config_.moe.expert_parallel_plan,
                "expert_overlay.plan");
        }
        else
        {
            material.moe.push_back({"expert_overlay.plan.enabled", "false"});
        }

        if (config_.moe.expert_overlay_runtime_plan)
        {
            appendExpertOverlayRuntimeFingerprintFields(
                material.moe,
                *config_.moe.expert_overlay_runtime_plan,
                "expert_overlay.runtime");
        }
        else
        {
            material.moe.push_back({"expert_overlay.runtime.enabled", "false"});
        }
    }

    IMoERuntimeTable *Qwen35MoEGraph::moeRuntimeTableForDevice(DeviceId device,
                                                               int prefill_token_capacity,
                                                               const std::string &key_suffix,
                                                               int num_layers_override,
                                                               bool register_decode_histogram)
    {
#if !defined(HAVE_ROCM) && !defined(HAVE_CUDA)
        (void)device;
        (void)prefill_token_capacity;
        (void)key_suffix;
        return nullptr;
#else
        // Device-routed grouped MoE (decode + grouped prefill) is supported on any
        // GPU backend: MoERuntimeTable mirrors its placement banks through the
        // generic IBackend abstraction (see MoERuntimeTable.cpp mirrorBackend),
        // so CUDA and ROCm are both valid here. Gating on is_rocm() previously
        // left CUDA without a runtime table, which forced every MoE routing/expert
        // decode stage into a non-capturable manual graph segment.
        const int table_layers = num_layers_override > 0 ? num_layers_override : config_.n_layers;
        if (!device.is_gpu() || config_.moe.num_experts <= 0 || config_.moe.top_k <= 0 || table_layers <= 0)
            return nullptr;

        const std::string key = key_suffix.empty()
                                    ? device.to_string()
                                    : device.to_string() + "#" + key_suffix;
        auto it = moe_runtime_tables_.find(key);
        if (it != moe_runtime_tables_.end())
        {
            if (prefill_token_capacity > 0)
                it->second->ensurePrefillRouteScratchCapacity(prefill_token_capacity);
            if (!key_suffix.empty() && key_suffix.rfind("mtp_depth", 0) == 0)
            {
                PerfStatsCollector::addCounter(
                    "mtp",
                    "moe_mtp_sidecar_runtime_table_reuses",
                    1.0,
                    "graph",
                    device.to_string(),
                    {{"key_suffix", key_suffix},
                     {"layers", std::to_string(table_layers)},
                     {"histogram_sync", "disabled"}});
            }
            return it->second.get();
        }

        DeviceMoERuntimeTable::Config table_config;
        table_config.device_id = device;
        table_config.num_layers = table_layers;
        table_config.num_experts = config_.moe.num_experts;
        table_config.top_k = config_.moe.top_k;
        table_config.mirror_to_device = true;
        table_config.prefill_token_capacity = std::max(0, prefill_token_capacity);

        auto table = std::make_unique<MoERuntimeTable>(table_config);
        IMoERuntimeTable *ptr = table.get();
        if (register_decode_histogram && config_.moe.decode_histogram)
        {
            auto *histogram = config_.moe.decode_histogram;
            histogram->registerRuntimeHistogramSync([ptr, histogram]()
                                                    { return ptr->syncDecodeHistogramToHost(*histogram); });
        }
        moe_runtime_tables_.emplace(key, std::move(table));
        if (!key_suffix.empty() && key_suffix.rfind("mtp_depth", 0) == 0)
        {
            PerfStatsCollector::addCounter(
                "mtp",
                "moe_mtp_sidecar_runtime_table_creations",
                1.0,
                "graph",
                device.to_string(),
                {{"key_suffix", key_suffix},
                 {"layers", std::to_string(table_layers)},
                 {"num_experts", std::to_string(config_.moe.num_experts)},
                 {"top_k", std::to_string(config_.moe.top_k)},
                 {"histogram_sync", register_decode_histogram
                                        ? "enabled"
                                        : "disabled"}});
        }
        return ptr;
#endif
    }

    // =========================================================================
    // Schema
    // =========================================================================

    GraphSchema Qwen35MoEGraph::getSchema() const
    {
        Qwen35MoESchemaFactory factory;
        GraphSchema schema = factory.createSchema();

        // Populate layer_template_names from config_.layer_types
        if (!config_.layer_types.empty())
        {
            schema.layer_template_names.resize(config_.n_layers);
            for (int i = 0; i < config_.n_layers; ++i)
            {
                schema.layer_template_names[i] = config_.layer_types[i];
            }
        }

        return schema;
    }

    // =========================================================================
    // Resolver Config (MoE buffer registration)
    // =========================================================================

    GraphResolverConfig Qwen35MoEGraph::getResolverConfig(int seq_len) const
    {
        // Start with base Qwen35 resolver config (GDN buffers, etc.)
        GraphResolverConfig config = Qwen35Graph::getResolverConfig(seq_len);

        // Add MoE-specific custom formulas
        int expert_intermediate = config_.moe.intermediate_size;
        int top_k = config_.moe.top_k;

        config.custom_formulas["moe_top_k"] = static_cast<size_t>(top_k);
        config.custom_formulas["moe_expert_intermediate"] = static_cast<size_t>(expert_intermediate);

        // Add MoE buffer name → BufferId mappings
        config.buffer_name_to_id["moe_expert_indices"] = BufferId::MOE_EXPERT_INDICES;
        config.buffer_name_to_id["moe_expert_weights"] = BufferId::MOE_EXPERT_WEIGHTS;
        config.buffer_name_to_id["moe_combined_output"] = BufferId::MOE_COMBINED_OUTPUT;
        config.buffer_name_to_id["moe_shared_expert_output"] = BufferId::MOE_SHARED_EXPERT_OUTPUT;
        config.buffer_name_to_id["moe_gate_scratch"] = BufferId::MOE_GATE_SCRATCH;
        config.buffer_name_to_id["moe_up_scratch"] = BufferId::MOE_UP_SCRATCH;

        LOG_DEBUG("[Qwen35MoEGraph::getResolverConfig] MoE formulas: "
                  << "moe_top_k=" << top_k
                  << ", moe_expert_intermediate=" << expert_intermediate);

        return config;
    }

    // =========================================================================
    // MoE FFN Graph Building
    // =========================================================================

    ComputeGraph Qwen35MoEGraph::buildFFNGraph(
        const LayerWeights &layer,
        ActivationBuffers &buffers,
        int layer_idx,
        int seq_len,
        int batch_size,
        DeviceId device)
    {
        // If this layer doesn't have MoE weights, fall back to dense FFN
        if (!layer.moe_gate || !layer.moe_gate_exps)
        {
            return Qwen35Graph::buildFFNGraph(layer, buffers, layer_idx, seq_len, batch_size, device);
        }

        ComputeGraph graph;
        const bool mtp_sidecar_context = mtp_graph_context_active_;
        const int mtp_depth_idx = mtp_graph_depth_idx_;
        std::string prefix = mtp_sidecar_context
                                 ? "MTP" + std::to_string(std::max(0, mtp_depth_idx)) + "_"
                                 : "layer" + std::to_string(layer_idx) + "_";
        std::string ffn_terminal;
        int total_tokens = batch_size * seq_len;
        auto overlay_runtime_plan = runtimePlanForGraph(config_);
        const auto &overlay_plan = overlay_runtime_plan
                                       ? overlay_runtime_plan->sourcePlanPtr()
                                       : config_.moe.expert_parallel_plan;
        auto domainContainsDevice = [](const MoEOverlayRuntimeDomain &domain, DeviceId candidate)
        {
            return std::any_of(domain.participants.begin(), domain.participants.end(),
                               [&](const MoEOverlayDomainParticipant &participant)
                               {
                                   return participant.local_device == candidate;
                               });
        };

        if (overlay_runtime_plan)
        {
            const auto &continuation_domain = overlay_runtime_plan->continuationDomain();
            DeviceId continuation_device = overlay_runtime_plan->continuationDevice();
            if (continuation_device.is_valid() && continuation_device != device &&
                !domainContainsDevice(continuation_domain, device))
            {
                LOG_DEBUG("[Qwen35MoEGraph] Layer " << layer_idx
                                                    << " using MoE overlay continuation_domain root device "
                                                    << continuation_device.to_string()
                                                    << " instead of caller device " << device.to_string());
                device = continuation_device;
            }
        }

        const bool overlay_requested = overlay_plan && overlay_plan->isTieredOverlay();
        const ExpertLayerPlacement *overlay_placement = overlay_requested
                                                            ? findExpertOverlayPlacement(*overlay_plan, layer_idx)
                                                            : nullptr;
        const bool use_expert_overlay = overlay_requested && overlay_placement &&
                                        isUsableExpertOverlayPlacement(*overlay_placement,
                                                                       *overlay_plan,
                                                                       config_.moe.num_experts,
                                                                       layer_idx);

        /*
         * Verifier batches are tiny (draft depth + bonus row), but the main-model
         * verifier is stateful: its row logits must match serial decode and the
         * accepted row may later publish KV/GDN/conv state.  Phase 9.8 therefore
         * only permits grouped verifier execution through decode-equivalent
         * M=2..4 stage/kernel contracts.  The router still emits serial-decode
         * top-k rows, and the shared expert uses GEMV verifier-row hooks rather
         * than the older MoE grouped prefill helper.
         */
        auto forceGpuSmallMMoESidecarPrefill = [&](DeviceId candidate)
        {
            return (candidate.is_cuda() || candidate.is_rocm()) &&
                   total_tokens > 1 &&
                   total_tokens <= 4 &&
                   mtp_sidecar_context;
        };
        auto forceGpuSmallMMainVerifierPrefill = [&](DeviceId candidate)
        {
            return (candidate.is_cuda() || candidate.is_rocm()) &&
                   total_tokens > 1 &&
                   total_tokens <= 4 &&
                   config_.compute_all_position_logits &&
                   !mtp_sidecar_context;
        };
        auto forceGroupedMoEVerifierPrefill = [&](DeviceId candidate)
        {
            return forceGpuSmallMMoESidecarPrefill(candidate) ||
                   forceGpuSmallMMainVerifierPrefill(candidate);
        };
        auto forceDecodeEquivalentMoERouting = [&](DeviceId candidate)
        {
            /*
             * The main verifier compares grouped all-position rows against
             * serial decode rows.  The expert FFN may still use the economical
             * grouped M=2..4 path, but the router must publish the same top-k
             * ids and weights that serial decode would have produced.  Keeping
             * routing and expert execution as separate decisions avoids the
             * previous ROCm drift where a correct grouped expert kernel was fed
             * slightly different all-position prefill routes.
             */
            return (candidate.is_cpu() || candidate.is_cuda() || candidate.is_rocm()) &&
                   total_tokens > 1 &&
                   total_tokens <= 4 &&
                   config_.compute_all_position_logits &&
                   !mtp_sidecar_context;
        };
        auto forceDecodeEquivalentMoEVerifier = [&](DeviceId candidate)
        {
            return (candidate.is_cpu() || candidate.is_cuda() || candidate.is_rocm()) &&
                   total_tokens > 1 &&
                   total_tokens <= 4 &&
                   config_.compute_all_position_logits &&
                   !mtp_sidecar_context &&
                   !forceGpuSmallMMainVerifierPrefill(candidate);
        };
        LayerWeightBindings layer_bindings = layerWeightBindingsForGraph(layer_idx);

        IMoERuntimeTable *moe_runtime_table = nullptr;
        const auto &rocm_env = debugEnv().rocm;
        auto forceGroupedSharedMoEVerifierPrefill = [&](DeviceId candidate)
        {
            return rocm_env.moe_grouped_prefill &&
                   forceGroupedMoEVerifierPrefill(candidate);
        };
        /*
         * MTP sidecars need their own persistent MoE metadata even when their
         * logical layer index aliases a main-model layer.  The runtime table
         * stores device-resident top-k route IDs, route weights, placement
         * banks, and prefill scratch pointers that graph replay can capture by
         * address.  Sharing those slots with ordinary decode would make an
         * accepted-state publication boundary able to invalidate a warm sidecar
         * graph without changing any graph shape.  Depth-scoped tables match the
         * vLLM-style contract: sidecar metadata is stable, persistent, and owned
         * by the sidecar graph fragment.
         */
        const bool use_mtp_runtime_table = mtp_sidecar_context;
        const int runtime_table_layers = use_mtp_runtime_table
                                             ? std::max(config_.n_layers, layer_idx + 1)
                                             : config_.n_layers;
        const std::string runtime_table_suffix = use_mtp_runtime_table
                                                     ? "mtp_depth" + std::to_string(mtp_depth_idx)
                                                     : std::string{};
        const bool register_runtime_histogram = !use_mtp_runtime_table;
        if (total_tokens == 1 && rocm_env.moe_grouped_decode && rocm_env.moe_device_routed_decode)
        {
            moe_runtime_table = moeRuntimeTableForDevice(
                device,
                0,
                runtime_table_suffix,
                runtime_table_layers,
                register_runtime_histogram);
        }
        else if (total_tokens > 1 &&
                 (rocm_env.moe_grouped_prefill ||
                  forceDecodeEquivalentMoERouting(device)))
        {
            // Fixed-topology grouped prefill consumes routing tensors directly.
            // Decode-equivalent verifier routing instead uses the same
            // runtime-table router as serial decode, so build the table even
            // when grouped prefill is disabled; otherwise the strict verifier
            // row proof would fail closed before reaching the rows under test.
            moe_runtime_table = moeRuntimeTableForDevice(
                device,
                0,
                runtime_table_suffix,
                runtime_table_layers,
                register_runtime_histogram);
        }

        // =====================================================================
        // Stage 1: Pre-FFN RMSNorm (fused with attention residual add)
        // =====================================================================
        {
            FusedResidualNormStage::Params fused_params;
            fused_params.device_id = device;
            fused_params.input = buffers.attn_proj;
            fused_params.residual = buffers.current_hidden;
            fused_params.gamma = layer.ffn_norm;
            fused_params.norm_output = buffers.normalized;
            fused_params.eps = config_.rms_norm_eps;
            fused_params.seq_len = total_tokens;
            fused_params.hidden_dim = config_.d_model;
            fused_params.input_buffer_id = buffers.idFor(BufferId::ATTN_PROJ);
            fused_params.residual_buffer_id = buffers.idFor(BufferId::HIDDEN_STATE);
            fused_params.norm_output_buffer_id = buffers.idFor(BufferId::NORMALIZED);

            graph.addNode(prefix + "ffn_norm",
                          ComputeStageFactory::createFusedResidualNorm(fused_params),
                          device);
            ffn_terminal = prefix + "ffn_norm";
        }

        // =====================================================================
        // Stage 2: MoE Routing (softmax top-k expert selection)
        // =====================================================================
        TensorBase *routing_indices = buffers.get(buffers.idFor(BufferId::MOE_EXPERT_INDICES));
        TensorBase *routing_weights = buffers.get(buffers.idFor(BufferId::MOE_EXPERT_WEIGHTS));

        {
            MoERoutingStage::Params route_params;
            route_params.device_id = device;
            route_params.input = buffers.normalized;
            route_params.seq_len = total_tokens;
            route_params.d_model = config_.d_model;
            route_params.gate_weights = layer.moe_gate;
            route_params.num_experts = config_.moe.num_experts;
            route_params.top_k = config_.moe.top_k;
            route_params.norm_topk_prob = config_.moe.norm_topk_prob;
            route_params.layer_idx = layer_idx;
            route_params.decode_histogram = mtp_sidecar_context ? nullptr : config_.moe.decode_histogram;
            route_params.moe_runtime_table = moe_runtime_table;
            route_params.force_grouped_verifier_prefill_for_decode =
                forceGroupedMoEVerifierPrefill(device);
            route_params.force_decode_equivalent_verifier_prefill =
                forceDecodeEquivalentMoERouting(device);
            route_params.output_indices = routing_indices;
            route_params.output_weights = routing_weights;
            route_params.input_buffer_id = buffers.idFor(BufferId::NORMALIZED);
            route_params.output_indices_buffer_id = buffers.idFor(BufferId::MOE_EXPERT_INDICES);
            route_params.output_weights_buffer_id = buffers.idFor(BufferId::MOE_EXPERT_WEIGHTS);

            graph.addNode(prefix + "moe_routing",
                          ComputeStageFactory::createMoERouting(route_params),
                          device);
            graph.addDependency(prefix + "moe_routing", prefix + "ffn_norm");
        }

        // =====================================================================
        // Stage 3: MoE Expert Compute (routed expert SwiGLU FFN)
        // =====================================================================
        TensorBase *moe_output = buffers.get(buffers.idFor(BufferId::MOE_COMBINED_OUTPUT));
        bool shared_gate_writes_combined_output = false;

        {
            // Infer expert intermediate size from weight shape
            int expert_intermediate = config_.moe.intermediate_size;
            if (expert_intermediate == 0 && layer.moe_gate_exps)
            {
                // gate_exps shape: [num_experts, intermediate, d_model] or rows=num_experts*intermediate
                size_t total_rows = layer.moe_gate_exps->rows();
                expert_intermediate = static_cast<int>(total_rows / config_.moe.num_experts);
            }

            auto makeExpertParams = [&](TensorBase *output,
                                        BufferId output_buffer_id,
                                        std::vector<bool> expert_mask,
                                        DeviceId stage_device)
            {
                MoEExpertComputeStage::Params expert_params;
                expert_params.device_id = stage_device;
                expert_params.input = buffers.normalized;
                expert_params.seq_len = total_tokens;
                expert_params.d_model = config_.d_model;
                expert_params.num_experts = config_.moe.num_experts;
                expert_params.top_k = config_.moe.top_k;
                expert_params.gate_exps = layer.moe_gate_exps;
                expert_params.up_exps = layer.moe_up_exps;
                expert_params.down_exps = layer.moe_down_exps;
                expert_params.expert_intermediate = expert_intermediate;
                expert_params.layer_idx = layer_idx;
                expert_params.routing_indices = routing_indices;
                expert_params.routing_weights = routing_weights;
                expert_params.routing_indices_buffer_id = buffers.idFor(BufferId::MOE_EXPERT_INDICES);
                expert_params.routing_weights_buffer_id = buffers.idFor(BufferId::MOE_EXPERT_WEIGHTS);
                expert_params.output = output;
                expert_params.output_buffer_id = output_buffer_id;
                expert_params.input_buffer_id = buffers.idFor(BufferId::NORMALIZED);
                expert_params.prepared_store = prepared_weight_store_;
                expert_params.expert_mask = std::move(expert_mask);
                expert_params.moe_runtime_table = moe_runtime_table;
                expert_params.force_grouped_verifier_prefill_for_decode =
                    forceGroupedMoEVerifierPrefill(stage_device);
                expert_params.force_decode_equivalent_verifier_prefill =
                    forceDecodeEquivalentMoEVerifier(stage_device);

                if (config_.moe.expert_mode == MoEExpertMode::ExpertParallel &&
                    expert_params.expert_mask.empty())
                {
                    expert_params.local_expert_start = config_.moe.local_expert_start;
                    expert_params.local_expert_count = config_.moe.local_expert_count;
                    if (expert_params.local_expert_count >= 0)
                    {
                        // LocalTP expert-parallel runners own a contiguous
                        // global expert-id range. Express that static range as
                        // the same explicit mask used by dynamic rebalance and
                        // overlay paths so GPU graph construction only
                        // requires resident GEMM engines for local experts,
                        // not the entire global MoE layer.
                        expert_params.expert_mask.assign(config_.moe.num_experts, false);
                        const int start = std::max(0, expert_params.local_expert_start);
                        const int end = std::min(config_.moe.num_experts,
                                                 start + expert_params.local_expert_count);
                        for (int expert = start; expert < end; ++expert)
                            expert_params.expert_mask[static_cast<size_t>(expert)] = true;
                    }
                }

                const int gpu_cache_experts = debugEnv().moe_rebalance.gpu_cache_experts_per_layer;
                if (expert_params.expert_mask.empty() && gpu_cache_experts > 0)
                {
                    expert_params.expert_mask.assign(config_.moe.num_experts, false);
                    for (int expert = 0; expert < config_.moe.num_experts; ++expert)
                        expert_params.expert_mask[expert] = !stage_device.is_gpu();
                    LOG_DEBUG("[Qwen35MoEGraph] Initial MoE GPU expert cache bootstrap mask: device="
                              << stage_device.to_string() << " layer=" << layer_idx
                              << " cpu_initial_owner=" << (!stage_device.is_gpu()));
                }

                // GPU scratch buffers
                expert_params.gate_scratch = buffers.get(buffers.idFor(BufferId::MOE_GATE_SCRATCH));
                expert_params.up_scratch = buffers.get(buffers.idFor(BufferId::MOE_UP_SCRATCH));

                return expert_params;
            };

            auto prepareExpertParams = [&](MoEExpertComputeStage::Params &expert_params,
                                           DeviceId stage_device,
                                           const std::string &placement_context = {},
                                           const std::string &registry_domain_name = {})
            {
                auto withPlacementContext = [&](const std::string &reason)
                {
                    if (placement_context.empty())
                        return reason;
                    return placement_context + ": " + reason;
                };

                // Extract per-expert 2D views from 3D packed tensors (required)
                if (!MoEExpertComputeStage::extractExpertViews(expert_params))
                {
                    LOG_ERROR("[Qwen35MoEGraph] Failed to extract expert views for layer " << layer_idx);
                    return false;
                }

                // Set expert_registry for dynamic rebalancing registry updates
                if (model_ctx_)
                {
                    auto weight_mgr = model_ctx_->concreteWeightManager();
                    if (weight_mgr)
                        expert_params.expert_registry = &weight_mgr->expertGemmRegistry();
                }

                // CPU prepares engines inline. GPU graph construction must consume the
                // unified pipeline registry and fail before execution if required
                // resident expert engines are absent.
                if (stage_device.is_gpu())
                {
                    const bool has_active_masked_expert = hasActiveExpertMask(expert_params.expert_mask);

                    auto weight_mgr = model_ctx_ ? model_ctx_->concreteWeightManager() : nullptr;
                    if (!weight_mgr)
                    {
                        if (expert_params.expert_mask.empty() || has_active_masked_expert)
                            failMissingGpuExpertGemmEngines(stage_device, layer_idx,
                                                            withPlacementContext(
                                                                "WeightManager unavailable (" +
                                                                describeMissingExpertGemmEngine(config_.moe.num_experts,
                                                                                                expert_params.expert_mask,
                                                                                                expert_params.prepared_gate_gemm,
                                                                                                expert_params.prepared_up_gemm,
                                                                                                expert_params.prepared_down_gemm) +
                                                                ")"));

                        expert_params.prepared_gate_gemm.assign(config_.moe.num_experts, nullptr);
                        expert_params.prepared_up_gemm.assign(config_.moe.num_experts, nullptr);
                        expert_params.prepared_down_gemm.assign(config_.moe.num_experts, nullptr);
                    }
                    else
                    {
                        const auto &registry = weight_mgr->expertGemmRegistry();
                        const bool domain_scoped = !registry_domain_name.empty();
                        const bool complete_layer = domain_scoped
                                                        ? registry.hasCompleteLayerInDomain(
                                                              registry_domain_name, stage_device, layer_idx, config_.moe.num_experts)
                                                        : registry.hasCompleteLayer(stage_device, layer_idx, config_.moe.num_experts);
                        const bool populated = domain_scoped
                                                   ? registry.populateExpertEnginesForDomain(
                                                         registry_domain_name, stage_device, layer_idx, config_.moe.num_experts,
                                                         expert_params.prepared_gate_gemm,
                                                         expert_params.prepared_up_gemm,
                                                         expert_params.prepared_down_gemm)
                                                   : registry.populateExpertEngines(stage_device, layer_idx, config_.moe.num_experts,
                                                                                    expert_params.prepared_gate_gemm,
                                                                                    expert_params.prepared_up_gemm,
                                                                                    expert_params.prepared_down_gemm);

                        if (expert_params.expert_mask.empty())
                        {
                            if (!complete_layer || !populated)
                                failMissingGpuExpertGemmEngines(
                                    stage_device, layer_idx,
                                    withPlacementContext("incomplete ExpertGemmRegistry entry (" +
                                                         describeMissingExpertGemmEngine(config_.moe.num_experts,
                                                                                         expert_params.expert_mask,
                                                                                         expert_params.prepared_gate_gemm,
                                                                                         expert_params.prepared_up_gemm,
                                                                                         expert_params.prepared_down_gemm) +
                                                         ")"));
                        }
                        else if (has_active_masked_expert)
                        {
                            for (int expert = 0; expert < config_.moe.num_experts; ++expert)
                            {
                                if (!expert_params.expert_mask[expert])
                                    continue;
                                if (expert_params.prepared_gate_gemm[expert] == nullptr ||
                                    expert_params.prepared_up_gemm[expert] == nullptr ||
                                    expert_params.prepared_down_gemm[expert] == nullptr)
                                    failMissingGpuExpertGemmEngines(
                                        stage_device, layer_idx,
                                        withPlacementContext("missing active masked expert engine (" +
                                                             describeMissingExpertGemmEngine(config_.moe.num_experts,
                                                                                             expert_params.expert_mask,
                                                                                             expert_params.prepared_gate_gemm,
                                                                                             expert_params.prepared_up_gemm,
                                                                                             expert_params.prepared_down_gemm) +
                                                             ")"));
                            }
                        }

                        LOG_DEBUG("[Qwen35MoEGraph] Layer " << layer_idx
                                                            << ": populated expert GEMM engines from registry"
                                                            << (domain_scoped ? " domain=" + registry_domain_name : std::string())
                                                            << " complete_layer=" << complete_layer
                                                            << " active_masked_expert=" << has_active_masked_expert);
                    }
                }
                else
                {
                    if (!MoEExpertComputeStage::prepareExpertGemmEngines(expert_params))
                        return false;
                }

                return true;
            };

            /*
             * The old combined verifier represented the shared expert as one
             * extra routed expert in the backend MoE prefill table; strict
             * full-continuation parity rejected that math.  The current combined
             * verifier is a safe composite owner instead: routed grouped prefill
             * stays routed, shared expert uses decode-equivalent GEMV-many, and
             * the normal shared gate-add kernel joins the branches.  Limit this
             * promotion to the proven single-device GPU verifier shape.
             */
            const bool can_combine_shared_verifier =
                !use_expert_overlay &&
                !mtp_sidecar_context &&
                config_.compute_all_position_logits &&
                forceGroupedSharedMoEVerifierPrefill(device) &&
                layer.shared_expert_gate &&
                layer.shared_expert_up &&
                layer.shared_expert_down &&
                layer.shared_expert_gate_inp &&
                buffers.attn_proj;

            if (overlay_requested && !use_expert_overlay)
            {
                LOG_WARN("[Qwen35MoEGraph] Expert overlay requested for layer " << layer_idx
                                                                                << " but no usable placement was found; using legacy routed expert path");
            }

            if (use_expert_overlay)
            {
                auto dispatch_output_lifetime = std::make_shared<MoEExpertDispatchOutput>();

                MoEExpertDispatchStage::Params dispatch_params;
                dispatch_params.device_id = DeviceId::cpu();
                dispatch_params.routing_indices = routing_indices;
                dispatch_params.routing_weights = routing_weights;
                dispatch_params.hidden = buffers.normalized;
                dispatch_params.routing_indices_buffer_id = buffers.idFor(BufferId::MOE_EXPERT_INDICES);
                dispatch_params.routing_weights_buffer_id = buffers.idFor(BufferId::MOE_EXPERT_WEIGHTS);
                dispatch_params.hidden_buffer_id = buffers.idFor(BufferId::NORMALIZED);
                dispatch_params.seq_len = total_tokens;
                dispatch_params.top_k = config_.moe.top_k;
                dispatch_params.d_model = config_.d_model;
                dispatch_params.continuation_domain = overlay_runtime_plan
                                                          ? overlay_runtime_plan->continuationDomain().name
                                                          : overlay_plan->continuation_domain;
                dispatch_params.transfer_mode = debugEnv().moe_expert_overlay.dense_transfer
                                                    ? MoEExpertTransferMode::DenseFullSequence
                                                    : MoEExpertTransferMode::Auto;
                dispatch_params.placement = *overlay_placement;
                dispatch_params.routed_tiers = overlay_plan->routed_tiers;
                dispatch_params.output_lifetime = dispatch_output_lifetime;

                const std::string dispatch_name = prefix + "moe_expert_dispatch";
                graph.addNode(dispatch_name,
                              ComputeStageFactory::createMoEExpertDispatch(dispatch_params),
                              DeviceId::cpu());
                graph.addDependency(dispatch_name, prefix + "moe_routing");

                auto owner_map_lifetime = std::make_shared<MoEExpertOwnerMap>(
                    MoEExpertOwnerMap::build(*overlay_plan));
                const int continuation_root_participant = continuationRootParticipant(*overlay_plan);
                const int participant_count = participantCountForGraphNativeOverlay(
                    *owner_map_lifetime,
                    continuation_root_participant);

                std::vector<std::shared_ptr<MoEOverlayCollectiveWorkspace>> participant_workspaces(
                    static_cast<size_t>(participant_count));
                for (auto &workspace : participant_workspaces)
                {
                    workspace = std::make_shared<MoEOverlayCollectiveWorkspace>();
                    workspace->ensureCapacity(static_cast<size_t>(std::max(total_tokens, 1)),
                                              static_cast<size_t>(std::max(total_tokens * config_.moe.top_k, 1)),
                                              config_.d_model,
                                              config_.moe.top_k,
                                              DeviceId::cpu());
                }

                MoEOverlayLocalSparseCollectiveContext::Config collective_config;
                collective_config.participant_count = participant_count;
                collective_config.slot_count = std::max<size_t>(
                    8,
                    overlay_plan->routed_tiers.size() * static_cast<size_t>(participant_count) * 4u + 8u);
                auto collective_context_lifetime = std::make_shared<MoEOverlayLocalSparseCollectiveContext>(
                    collective_config);

                std::string last_return_reduce;
                bool first_return_scatter = true;

                for (size_t tier_index = 0; tier_index < overlay_plan->routed_tiers.size(); ++tier_index)
                {
                    const auto &tier = overlay_plan->routed_tiers[tier_index];
                    auto tier_mask = expertMaskForTier(*overlay_placement,
                                                       config_.moe.num_experts,
                                                       static_cast<int>(tier_index));
                    if (!hasActiveExpertMask(tier_mask))
                    {
                        LOG_DEBUG("[Qwen35MoEGraph] Skipping inactive MoE expert overlay tier "
                                  << tier.name << " for layer " << layer_idx);
                        continue;
                    }

                    std::vector<int> target_participants = owner_map_lifetime->participantIdsForTier(
                        static_cast<int>(tier_index));
                    const bool local_tp_replicated_tier =
                        isLocalTPReplicatedExpertsTier(*overlay_plan, tier);
                    const int graph_local_participant =
                        local_tp_replicated_tier
                            ? participantIdForTierDevice(
                                  *owner_map_lifetime,
                                  static_cast<int>(tier_index),
                                  device)
                            : -1;
                    const bool compute_replicated_tier_on_graph_local_participant =
                        graph_local_participant >= 0;
                    if (compute_replicated_tier_on_graph_local_participant)
                    {
                        target_participants = {graph_local_participant};
                    }

                    for (const int target_participant : target_participants)
                    {
                        auto participant_mask =
                            owner_map_lifetime->expertMaskForParticipant(
                                layer_idx,
                                target_participant,
                                config_.moe.num_experts);
                        if (!hasActiveExpertMask(participant_mask))
                            continue;

                        const DeviceId target_device = participantDeviceForGraphNativeOverlay(
                            *owner_map_lifetime,
                            target_participant);
                        if (target_device.is_gpu() && target_device != device)
                        {
                            throw std::runtime_error(
                                "Qwen35 MoE graph-native overlay would execute GPU expert participant " +
                                std::to_string(target_participant) + " on " +
                                target_device.to_string() + " from graph device " +
                                device.to_string() +
                                ". LocalTP ReplicatedExperts GPU domains must lower only the graph-local participant; "
                                "other cross-device GPU expert execution requires a real participant/domain executor.");
                        }
                        const std::string participant_suffix =
                            nodeSuffixForTier(tier, static_cast<int>(tier_index)) +
                            "_p" + std::to_string(target_participant);

                        auto target_dispatch_inbound = std::make_shared<MoEOverlaySparseRows>(
                            participant_workspaces[static_cast<size_t>(target_participant)]->dispatchReceive(
                                layer_idx,
                                static_cast<int>(tier_index)));
                        const MoEOverlayCollectiveKey dispatch_key = graphNativeMoEKey(
                            layer_idx,
                            static_cast<int>(tier_index),
                            target_participant,
                            MoEOverlayCollectiveDirection::Dispatch,
                            mtp_sidecar_context,
                            mtp_depth_idx);

                        std::string previous_dispatch_node;
                        std::string target_dispatch_node;
                        for (const int source_participant : participantsWithLast(participant_count, target_participant))
                        {
                            auto inbound_lifetime = source_participant == target_participant
                                                        ? target_dispatch_inbound
                                                        : std::make_shared<MoEOverlaySparseRows>(
                                                              participant_workspaces[static_cast<size_t>(source_participant)]->dispatchReceive(
                                                                  layer_idx,
                                                                  static_cast<int>(tier_index)));

                            MoESparseDispatchStage::Params sparse_dispatch_params;
                            sparse_dispatch_params.device_id = DeviceId::cpu();
                            sparse_dispatch_params.collective_context_lifetime = collective_context_lifetime;
                            sparse_dispatch_params.workspace_lifetime = participant_workspaces[static_cast<size_t>(source_participant)];
                            sparse_dispatch_params.key = dispatch_key;
                            sparse_dispatch_params.source_participant = source_participant;
                            sparse_dispatch_params.target_participant = target_participant;
                            sparse_dispatch_params.seq_len = total_tokens;
                            sparse_dispatch_params.top_k = config_.moe.top_k;
                            sparse_dispatch_params.d_model = config_.d_model;
                            sparse_dispatch_params.tier_index = static_cast<int>(tier_index);
                            sparse_dispatch_params.replicated_hidden_export = true;
                            sparse_dispatch_params.logical_continuation_root_participant = continuation_root_participant;
                            sparse_dispatch_params.manual_boundary_requires_collective_completion =
                                source_participant == target_participant;
                            sparse_dispatch_params.inbound_rows_lifetime = inbound_lifetime;
                            if (source_participant == continuation_root_participant)
                            {
                                sparse_dispatch_params.hidden = buffers.normalized;
                                sparse_dispatch_params.routing_indices = routing_indices;
                                sparse_dispatch_params.routing_weights = routing_weights;
                                sparse_dispatch_params.hidden_buffer_id = buffers.idFor(BufferId::NORMALIZED);
                                sparse_dispatch_params.routing_indices_buffer_id = buffers.idFor(BufferId::MOE_EXPERT_INDICES);
                                sparse_dispatch_params.routing_weights_buffer_id = buffers.idFor(BufferId::MOE_EXPERT_WEIGHTS);
                                sparse_dispatch_params.dispatch_output_lifetime = dispatch_output_lifetime;
                            }

                            const std::string sparse_dispatch_name = prefix + "moe_sparse_dispatch_" +
                                                                     participant_suffix +
                                                                     "_from_p" + std::to_string(source_participant);
                            graph.addNode(sparse_dispatch_name,
                                          ComputeStageFactory::createMoESparseDispatch(sparse_dispatch_params),
                                          DeviceId::cpu());
                            graph.addDependency(sparse_dispatch_name,
                                                previous_dispatch_node.empty() ? dispatch_name : previous_dispatch_node);
                            previous_dispatch_node = sparse_dispatch_name;
                            if (source_participant == target_participant)
                                target_dispatch_node = sparse_dispatch_name;
                        }

                        auto local_output_lifetime = std::make_shared<MoEOverlayReturnRows>(
                            participant_workspaces[static_cast<size_t>(target_participant)]->localExpertOutput(
                                layer_idx,
                                static_cast<int>(tier_index)));

                        MoELocalExpertStage::Params local_params;
                        local_params.device_id = target_device;
                        local_params.input_rows_lifetime = target_dispatch_inbound;
                        local_params.output_rows_lifetime = local_output_lifetime;
                        local_params.workspace_lifetime = participant_workspaces[static_cast<size_t>(target_participant)];
                        local_params.gate_exps = layer.moe_gate_exps;
                        local_params.up_exps = layer.moe_up_exps;
                        local_params.down_exps = layer.moe_down_exps;
                        local_params.num_experts = config_.moe.num_experts;
                        local_params.top_k = config_.moe.top_k;
                        local_params.d_model = config_.d_model;
                        local_params.expert_intermediate = expert_intermediate;
                        local_params.layer_idx = layer_idx;
                        local_params.expert_mask = std::move(participant_mask);
                        local_params.prepared_store = prepared_weight_store_;
                        local_params.runtime_participant_index = target_participant;
                        if (model_ctx_)
                        {
                            auto weight_mgr = model_ctx_->concreteWeightManager();
                            if (weight_mgr)
                            {
                                auto &registry = weight_mgr->expertGemmRegistry();
                                local_params.expert_registry = &registry;
                                bool populated = false;
                                auto has_active_prepared_engines = [&]() -> bool
                                {
                                    if (local_params.prepared_gate_gemm.size() !=
                                            static_cast<size_t>(config_.moe.num_experts) ||
                                        local_params.prepared_up_gemm.size() !=
                                            static_cast<size_t>(config_.moe.num_experts) ||
                                        local_params.prepared_down_gemm.size() !=
                                            static_cast<size_t>(config_.moe.num_experts))
                                    {
                                        return false;
                                    }

                                    for (int expert = 0; expert < config_.moe.num_experts; ++expert)
                                    {
                                        if (!local_params.expert_mask[static_cast<size_t>(expert)])
                                            continue;
                                        if (!local_params.prepared_gate_gemm[static_cast<size_t>(expert)] ||
                                            !local_params.prepared_up_gemm[static_cast<size_t>(expert)] ||
                                            !local_params.prepared_down_gemm[static_cast<size_t>(expert)])
                                        {
                                            return false;
                                        }
                                        populated = true;
                                    }
                                    return populated;
                                };
                                if (const auto *participant = owner_map_lifetime->participantForId(target_participant))
                                {
                                    (void)registry.populateExpertEnginesForParticipant(
                                        tier.domain,
                                        target_device,
                                        participant->world_rank_known ? participant->world_rank : -1,
                                        participant->domain_participant_index,
                                        layer_idx,
                                        config_.moe.num_experts,
                                        local_params.prepared_gate_gemm,
                                        local_params.prepared_up_gemm,
                                        local_params.prepared_down_gemm);
                                    populated = has_active_prepared_engines();
                                }
                                if (!populated)
                                {
                                    (void)registry.populateExpertEnginesForDomain(
                                        tier.domain,
                                        target_device,
                                        layer_idx,
                                        config_.moe.num_experts,
                                        local_params.prepared_gate_gemm,
                                        local_params.prepared_up_gemm,
                                        local_params.prepared_down_gemm);
                                }
                            }
                        }

                        const std::string local_name = prefix + "moe_local_expert_" + participant_suffix;
                        graph.addNode(local_name,
                                      ComputeStageFactory::createMoELocalExpert(local_params),
                                      target_device);
                        graph.addDependency(local_name,
                                            target_dispatch_node.empty() ? previous_dispatch_node : target_dispatch_node);

                        const MoEOverlayCollectiveKey return_key = graphNativeMoEKey(
                            layer_idx,
                            static_cast<int>(tier_index),
                            target_participant,
                            MoEOverlayCollectiveDirection::ReturnReduce,
                            mtp_sidecar_context,
                            mtp_depth_idx);
                        std::string previous_return_node = last_return_reduce;
                        std::string root_return_node;
                        for (const int source_participant : participantsWithLast(participant_count, continuation_root_participant))
                        {
                            std::shared_ptr<const MoEOverlayReturnRows> outbound_lifetime;
                            if (source_participant == target_participant)
                            {
                                outbound_lifetime = local_output_lifetime;
                            }
                            else
                            {
                                outbound_lifetime = std::make_shared<MoEOverlayReturnRows>(
                                    participant_workspaces[static_cast<size_t>(source_participant)]->localExpertOutput(
                                        layer_idx,
                                        static_cast<int>(tier_index)));
                            }

                            auto inbound_lifetime = std::make_shared<MoEOverlayReturnRows>(
                                participant_workspaces[static_cast<size_t>(source_participant)]->returnReceive(
                                    layer_idx,
                                    static_cast<int>(tier_index)));

                            MoESparseReturnReduceStage::Params return_params;
                            return_params.device_id = DeviceId::cpu();
                            return_params.collective_context_lifetime = collective_context_lifetime;
                            return_params.workspace_lifetime = participant_workspaces[static_cast<size_t>(source_participant)];
                            return_params.key = return_key;
                            return_params.source_participant = source_participant;
                            return_params.target_participant = continuation_root_participant;
                            return_params.outbound_rows_lifetime = outbound_lifetime;
                            return_params.inbound_rows_lifetime = inbound_lifetime;
                            return_params.dense_output = moe_output;
                            return_params.dense_output_buffer_id = buffers.idFor(BufferId::MOE_COMBINED_OUTPUT);
                            return_params.seq_len = total_tokens;
                            return_params.d_model = config_.d_model;
                            return_params.clear_output_before_scatter =
                                source_participant == continuation_root_participant && first_return_scatter;
                            return_params.manual_boundary_requires_collective_completion =
                                source_participant == continuation_root_participant;

                            const std::string return_name = prefix + "moe_sparse_return_reduce_" +
                                                            participant_suffix +
                                                            "_from_p" + std::to_string(source_participant);
                            graph.addNode(return_name,
                                          ComputeStageFactory::createMoESparseReturnReduce(return_params),
                                          DeviceId::cpu());
                            graph.addDependency(return_name, local_name);
                            if (!previous_return_node.empty())
                                graph.addDependency(return_name, previous_return_node);
                            previous_return_node = return_name;
                            if (source_participant == continuation_root_participant)
                                root_return_node = return_name;
                        }

                        if (!root_return_node.empty())
                        {
                            std::string tier_terminal = root_return_node;
                            if (compute_replicated_tier_on_graph_local_participant && needsTPAllreduce())
                            {
                                const size_t allreduce_count =
                                    static_cast<size_t>(total_tokens) * static_cast<size_t>(config_.d_model);
                                const std::string ar_name = prefix + "moe_sparse_return_reduce_" +
                                                            participant_suffix + "_allreduce";
                                auto allreduce_stage = createTPAllreduceStage(
                                    moe_output,
                                    allreduce_count,
                                    device,
                                    layer_idx,
                                    /*is_attention=*/false,
                                    ar_name,
                                    buffers.idFor(BufferId::MOE_COMBINED_OUTPUT));
                                if (allreduce_stage)
                                {
                                    graph.addNode(ar_name, std::move(allreduce_stage), device);
                                    graph.addDependency(ar_name, root_return_node);
                                    tier_terminal = ar_name;
                                }
                            }

                            last_return_reduce = tier_terminal;
                            first_return_scatter = false;
                        }
                    }
                }

                if (last_return_reduce.empty())
                {
                    throw std::runtime_error("Qwen35 MoE graph-native overlay produced no local expert participants for layer " +
                                             std::to_string(layer_idx));
                }
                ffn_terminal = last_return_reduce;
            }
            else
            {
                auto expert_params = makeExpertParams(can_combine_shared_verifier ? buffers.attn_proj : moe_output,
                                                      can_combine_shared_verifier
                                                          ? buffers.idFor(BufferId::ATTN_PROJ)
                                                          : buffers.idFor(BufferId::MOE_COMBINED_OUTPUT),
                                                      {},
                                                      device);
                if (can_combine_shared_verifier)
                {
                    expert_params.combine_shared_expert_in_verifier = true;
                    expert_params.shared_gate_w = layer.shared_expert_gate;
                    expert_params.shared_up_w = layer.shared_expert_up;
                    expert_params.shared_down_w = layer.shared_expert_down;
                    expert_params.shared_gate_inp = layer.shared_expert_gate_inp;
                    expert_params.prepared_shared_ref_gate = preparedRefForGraphWeight(
                        layer_bindings.shared_expert_gate, device);
                    expert_params.prepared_shared_ref_up = preparedRefForGraphWeight(
                        layer_bindings.shared_expert_up, device);
                    expert_params.prepared_shared_ref_down = preparedRefForGraphWeight(
                        layer_bindings.shared_expert_down, device);
                }
                if (!prepareExpertParams(expert_params, device))
                {
                    throw std::runtime_error(
                        "Qwen35 MoE graph failed to prepare expert parameters for layer " +
                        std::to_string(layer_idx) + " on " + device.to_string());
                }

                if (device.is_gpu() &&
                    total_tokens == 1 &&
                    moe_runtime_table &&
                    debugEnv().rocm.moe_grouped_decode &&
                    debugEnv().rocm.moe_device_routed_decode)
                {
                    const bool full_local_decode =
                        expert_params.local_expert_start == 0 &&
                        (expert_params.local_expert_count < 0 ||
                         expert_params.local_expert_count == config_.moe.num_experts) &&
                        expert_params.replica_set.num_replicated == 0 &&
                        allExpertsEnabled(expert_params.expert_mask, config_.moe.num_experts);
                    if (!full_local_decode)
                    {
                        throw std::runtime_error(
                            "Qwen35 MoE GPU decode runtime table requires full local expert "
                            "ownership for layer " +
                            std::to_string(layer_idx) + " on " + device.to_string());
                    }

                    if (!initializeFullLocalDecodeRuntimeTable(
                            moe_runtime_table,
                            layer_idx,
                            config_.moe.num_experts,
                            config_.moe.top_k,
                            config_.d_model,
                            expert_intermediate,
                            expert_params.prepared_gate_gemm,
                            expert_params.prepared_up_gemm,
                            expert_params.prepared_down_gemm,
                            nullptr,
                            "single-device GPU decode graph build"))
                    {
                        throw std::runtime_error(
                            "Qwen35 MoE graph failed to initialize device decode "
                            "runtime table for layer " +
                            std::to_string(layer_idx) + " on " + device.to_string());
                    }
                }

                graph.addNode(prefix + "moe_expert_ffn",
                              ComputeStageFactory::createMoEExpertCompute(expert_params),
                              device);
                graph.addDependency(prefix + "moe_expert_ffn", prefix + "moe_routing");
                ffn_terminal = prefix + "moe_expert_ffn";
                if (can_combine_shared_verifier)
                    shared_gate_writes_combined_output = true;

                // Qwen35 MoE expert weights are normally replicated, so every rank
                // computes the full routed-expert contribution. Only allreduce this
                // path when an explicit expert range/mask makes the output partial.
                const bool routed_expert_output_is_partial =
                    expert_params.local_expert_count >= 0 || !expert_params.expert_mask.empty();
                if (routed_expert_output_is_partial && needsTPAllreduce())
                {
                    size_t allreduce_count = static_cast<size_t>(total_tokens) * static_cast<size_t>(config_.d_model);
                    std::string ar_name = prefix + "moe_expert_allreduce";
                    auto allreduce_stage = createTPAllreduceStage(
                        moe_output, allreduce_count, device, layer_idx,
                        /*is_attention=*/false, ar_name, buffers.idFor(BufferId::MOE_COMBINED_OUTPUT));
                    if (allreduce_stage)
                    {
                        graph.addNode(ar_name, std::move(allreduce_stage), device);
                        graph.addDependency(ar_name, prefix + "moe_expert_ffn");
                        ffn_terminal = ar_name;
                    }
                }
            }
        }

        // =====================================================================
        // Stage 4: Shared Expert FFN (always-active dense SwiGLU)
        // =====================================================================
        TensorBase *shared_output = buffers.get(buffers.idFor(BufferId::MOE_SHARED_EXPERT_OUTPUT));
        std::string shared_ffn_last; // Track last shared expert stage (empty if no shared expert)

        if (!shared_gate_writes_combined_output &&
            layer.shared_expert_gate && layer.shared_expert_up && layer.shared_expert_down && shared_output)
        {
            DeviceId shared_device = device;
            if (overlay_runtime_plan)
            {
                const auto &continuation_domain = overlay_runtime_plan->continuationDomain();
                const auto &shared_domain = overlay_runtime_plan->sharedExpertDomain();
                shared_device = overlay_runtime_plan->sharedExpertDeviceForMVP(layer_idx);
                if (domainContainsDevice(shared_domain, device))
                    shared_device = device;

                LOG_DEBUG("[Qwen35MoEGraph] Layer " << layer_idx
                                                    << " shared expert uses domain " << shared_domain.name
                                                    << " on " << shared_device.to_string()
                                                    << "; final output returns to continuation domain "
                                                    << continuation_domain.name << " on " << device.to_string()
                                                    << " via MoE combine");
            }

            // Always use the actual weight dimensions (already sharded for TP).
            // config_.moe.shared_intermediate_size is the FULL size from metadata,
            // but when TP is active, gate/up weights have rows() == intermediate/tp_degree.
            int shared_intermediate = static_cast<int>(layer.shared_expert_gate->rows());

            SharedExpertFFNStage::Params shared_params;
            shared_params.device_id = shared_device;
            shared_params.input = buffers.normalized;
            shared_params.gate_w = layer.shared_expert_gate;
            shared_params.up_w = layer.shared_expert_up;
            shared_params.down_w = layer.shared_expert_down;
            shared_params.output = shared_output;
            shared_params.seq_len = total_tokens;
            shared_params.d_model = config_.d_model;
            shared_params.intermediate = shared_intermediate;
            shared_params.input_buffer_id = buffers.idFor(BufferId::NORMALIZED);
            shared_params.output_buffer_id = buffers.idFor(BufferId::MOE_SHARED_EXPERT_OUTPUT);
            /*
             * Shared-expert verifier rows are independent of top-k routing.
             * The routed+shared single-table shortcut remains disabled, but
             * the standalone shared expert now has a strict all-codebook
             * M=2..4 verifier proof on CUDA and ROCm.  Keep it wired through
             * the grouped verifier route whenever the backend grouped-prefill
             * capability is enabled, and let routing conservatism stay with
             * the router/routed-expert stages.
             */
            const bool shared_grouped_verifier_prefill =
                forceGroupedSharedMoEVerifierPrefill(shared_device);
            shared_params.force_grouped_verifier_prefill_for_decode =
                shared_grouped_verifier_prefill;
            shared_params.force_decode_equivalent_verifier_prefill =
                !shared_grouped_verifier_prefill &&
                forceDecodeEquivalentMoEVerifier(shared_device);
            shared_params.prepared_ref_gate = preparedRefForGraphWeight(
                layer_bindings.shared_expert_gate, shared_device);
            shared_params.prepared_ref_up = preparedRefForGraphWeight(
                layer_bindings.shared_expert_up, shared_device);
            shared_params.prepared_ref_down = preparedRefForGraphWeight(
                layer_bindings.shared_expert_down, shared_device);
            shared_params.prepared_store = prepared_weight_store_;

            graph.addNode(prefix + "shared_expert_ffn",
                          ComputeStageFactory::createSharedExpertFFN(shared_params),
                          shared_device);
            graph.addDependency(prefix + "shared_expert_ffn", prefix + "ffn_norm");
            const bool main_verifier_rows =
                !mtp_sidecar_context &&
                config_.compute_all_position_logits &&
                total_tokens > 1 &&
                total_tokens <= 4;
            /*
             * The accepted GPU shared-verifier route is the standalone
             * GEMV-many path inside SharedExpertFFNStage.  It does not touch
             * IMoEKernel grouped-prefill scratch, so it can remain a true graph
             * sibling of the routed expert branch.  Any route that still uses
             * row-serial replay or the backend MoE helper keeps the conservative
             * dependency until it has its own branch-scoped workspace proof.
             */
            const bool shared_verifier_owns_branch_local_math =
                main_verifier_rows && shared_grouped_verifier_prefill;
            if (main_verifier_rows &&
                !shared_verifier_owns_branch_local_math &&
                !ffn_terminal.empty())
            {
                /*
                 * Phase 9.8 correctness guard: only the standalone grouped
                 * shared-verifier route has strict branch-local ownership.  If
                 * this graph ever falls back to the row-serial verifier helper,
                 * the shared branch temporarily re-enters normal decode and can
                 * touch backend MoE bridge state.  Keep that path serialized
                 * until a dedicated branch-scoped workspace proof exists.
                 */
                graph.addDependency(prefix + "shared_expert_ffn", ffn_terminal);
            }
            shared_ffn_last = prefix + "shared_expert_ffn";

            // Allreduce after shared expert down projection (InputParallel sharding)
            if (needsTPAllreduce())
            {
                size_t allreduce_count = static_cast<size_t>(total_tokens) * static_cast<size_t>(config_.d_model);
                std::string ar_name = prefix + "shared_expert_allreduce";
                auto allreduce_stage = createTPAllreduceStage(
                    shared_output, allreduce_count, shared_device, layer_idx,
                    /*is_attention=*/false, ar_name, buffers.idFor(BufferId::MOE_SHARED_EXPERT_OUTPUT));
                if (allreduce_stage)
                {
                    graph.addNode(ar_name, std::move(allreduce_stage), shared_device);
                    graph.addDependency(ar_name, prefix + "shared_expert_ffn");
                    shared_ffn_last = ar_name;
                }
            }

            // Stage 4b: Sigmoid gate on shared expert output
            if (layer.shared_expert_gate_inp)
            {
                const bool can_fuse_gate_and_combine =
                    !overlay_runtime_plan && !needsTPAllreduce() && shared_device == device &&
                    moe_output && buffers.attn_proj;

                SharedExpertGateStage::Params gate_params;
                gate_params.device_id = shared_device;
                gate_params.input = buffers.normalized;
                gate_params.gate_inp = layer.shared_expert_gate_inp;
                gate_params.shared_output = shared_output;
                gate_params.seq_len = total_tokens;
                gate_params.d_model = config_.d_model;
                gate_params.input_buffer_id = buffers.idFor(BufferId::NORMALIZED);
                gate_params.output_buffer_id = buffers.idFor(BufferId::MOE_SHARED_EXPERT_OUTPUT);
                if (can_fuse_gate_and_combine)
                {
                    gate_params.routed_residual = moe_output;
                    gate_params.combined_output = buffers.attn_proj;
                    gate_params.residual_buffer_id = buffers.idFor(BufferId::MOE_COMBINED_OUTPUT);
                    gate_params.combined_output_buffer_id = buffers.idFor(BufferId::ATTN_PROJ);
                }

                graph.addNode(prefix + "shared_expert_gate",
                              ComputeStageFactory::createSharedExpertGate(gate_params),
                              shared_device);
                graph.addDependency(prefix + "shared_expert_gate", shared_ffn_last);
                if (can_fuse_gate_and_combine)
                {
                    // The fused epilogue consumes both the shared-expert output
                    // and the routed-expert output, so it is the final MoE FFN
                    // producer for this single-device layer.
                    graph.addDependency(prefix + "shared_expert_gate", ffn_terminal);
                    shared_gate_writes_combined_output = true;
                }
                shared_ffn_last = prefix + "shared_expert_gate";
                if (shared_gate_writes_combined_output)
                    ffn_terminal = prefix + "shared_expert_gate";
            }
        }

        // =====================================================================
        // Stage 5: Combine expert output + shared expert output → attn_proj
        // =====================================================================
        // The combined MoE output goes to attn_proj so that the next layer's
        // FusedResidualNormStage handles the residual add automatically.
        {
            if (shared_gate_writes_combined_output)
            {
                // SharedExpertGateStage already wrote the combined MoE output to
                // ATTN_PROJ, so no standalone residual-add combine node is needed.
            }
            else if (!shared_ffn_last.empty() && shared_output)
            {
                // Add expert_output + shared_expert_output → attn_proj
                ResidualAddStage::Params add_params;
                add_params.device_id = device;
                add_params.input = shared_output;
                add_params.residual = moe_output;
                add_params.output = buffers.attn_proj;
                add_params.num_elements = static_cast<size_t>(total_tokens) * static_cast<size_t>(config_.d_model);
                add_params.input_buffer_id = buffers.idFor(BufferId::MOE_SHARED_EXPERT_OUTPUT);
                add_params.residual_buffer_id = buffers.idFor(BufferId::MOE_COMBINED_OUTPUT);
                add_params.output_buffer_id = buffers.idFor(BufferId::ATTN_PROJ);

                graph.addNode(prefix + "moe_combine",
                              ComputeStageFactory::createResidualAdd(add_params),
                              device);
                graph.addDependency(prefix + "moe_combine", ffn_terminal);
                graph.addDependency(prefix + "moe_combine", shared_ffn_last);
                ffn_terminal = prefix + "moe_combine";
            }
            else
            {
                // No shared expert — copy expert output to attn_proj directly
                // (or if MOE_COMBINED_OUTPUT IS attn_proj, this is a no-op)
                if (moe_output != buffers.attn_proj)
                {
                    ResidualAddStage::Params copy_params;
                    copy_params.device_id = device;
                    copy_params.input = moe_output;
                    copy_params.residual = nullptr;
                    copy_params.output = buffers.attn_proj;
                    copy_params.num_elements = static_cast<size_t>(total_tokens) * static_cast<size_t>(config_.d_model);
                    copy_params.input_buffer_id = buffers.idFor(BufferId::MOE_COMBINED_OUTPUT);
                    copy_params.output_buffer_id = buffers.idFor(BufferId::ATTN_PROJ);

                    graph.addNode(prefix + "moe_combine",
                                  ComputeStageFactory::createResidualAdd(copy_params),
                                  device);
                    graph.addDependency(prefix + "moe_combine", ffn_terminal);
                    ffn_terminal = prefix + "moe_combine";
                }
            }
        }

        // =====================================================================
        // Stage 6: Explicit residual (last layer only)
        // =====================================================================
        const bool skip_ffn_residual = (layer_idx < config_.pp_layer_offset + config_.n_layers - 1);
        if (!skip_ffn_residual)
        {
            ResidualAddStage::Params res_params;
            res_params.device_id = device;
            res_params.input = buffers.attn_proj;
            res_params.residual = buffers.current_hidden;
            res_params.output = buffers.current_hidden;
            res_params.num_elements = static_cast<size_t>(total_tokens) * static_cast<size_t>(config_.d_model);
            res_params.input_buffer_id = buffers.idFor(BufferId::ATTN_PROJ);
            res_params.residual_buffer_id = buffers.idFor(BufferId::HIDDEN_STATE);
            res_params.output_buffer_id = buffers.idFor(BufferId::HIDDEN_STATE);

            graph.addNode(prefix + "ffn_residual",
                          ComputeStageFactory::createResidualAdd(res_params),
                          device);
            graph.addDependency(prefix + "ffn_residual", ffn_terminal);
            ffn_terminal = prefix + "ffn_residual";
        }

        graph.setTerminalNode(ffn_terminal);
        return graph;
    }

} // namespace llaminar2
