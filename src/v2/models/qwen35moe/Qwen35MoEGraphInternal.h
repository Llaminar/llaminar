/**
 * @file Qwen35MoEGraphInternal.h
 * @brief Shared includes, helpers, and FFN lowering context for Qwen35MoEGraph translation units.
 */
#pragma once

#include <functional>
#include "../../execution/moe/DeviceMoECommittedMaintenanceScope.h"
#include "execution/moe/MoEOverlayHostServiceContract.h"
#include "execution/moe/MoELocalTPMaintenanceDomain.h"
#include "Qwen35MoEGraph.h"
#include "models/qwen4exp/Qwen4ExpConfig.h"
#ifdef HAVE_CUDA
#include "backends/cuda/CUDAHardwareMetadata.h"
#endif
#ifdef HAVE_ROCM
#include "backends/rocm/ROCmHardwareMetadata.h"
#endif
#include "execution/compute_stages/stages/MoEIntermediateShardPipeline.h"
#include "execution/local_execution/graph/HeterogeneousOverlaySection.h"
#include "execution/moe/MoEIntermediateShardDescriptorDeclaration.h"
#include "execution/moe/MoEIntermediateShardRuntimeBinding.h"
#include "execution/moe/MoEIntermediateShardStaticGraphAdmission.h"
#include "models/qwen4exp/Qwen4ExpRetainedM1HCFFNPolicy.h"
#include "execution/moe/MoEOverlayActivationBatchInventory.h"
#include "execution/moe/MoEOverlayPreparedGraphBank.h"
#include "execution/moe/MoEGraphOwnerProjection.h"
#include "execution/moe/MoECPUExpertWorkspaceSource.h"
#include "planning/ModelMemoryProfile.h"
#include "planning/PhysicalMemoryAuthority.h"
#include "execution/moe/MoEExpertOverlayPreparationPlan.h"
#include "execution/moe/MoEExpertOverlayGraphScope.h"
#include "execution/moe/MoEProjectionRuntimeBinding.h"
#include "execution/moe/MoEOverlayNoPeerResidency.h"
#include "execution/compute_stages/stages/MoEProjectionPipeline.h"
#include "execution/compute_stages/stages/MoERoutingPipeline.h"
#include "loaders/PreparedWeightStore.h"
#include "execution/moe/DeviceMoEExpertDescriptorBuilder.h"
#include "Qwen35MoESchema.h"
#include "../../collective/ILocalTPContext.h"
#include "../../collective/IGlobalTPContext.h"
#include "../../interfaces/IMPIContext.h"
#include "../../utils/Logger.h"
#include "../../execution/compute_stages/ComputeStageFactory.h"
#include "../../execution/compute_stages/stages/MoEExpertDispatchStage.h"
#include "../../execution/compute_stages/stages/MoERoutingStage.h"
#include "../../execution/compute_stages/stages/MoEExpertComputeStage.h"
#include "../../execution/compute_stages/stages/MoEOverlayActivationPacketStages.h"
#include "../../execution/compute_stages/stages/HiddenStateRowSelectStage.h"
#include "../../execution/compute_stages/stages/MoELocalExpertStage.h"
#include "../../execution/compute_stages/stages/MoECPUExpertShardRouteStage.h"
#include "../../execution/moe/ExpertShardSourceSelection.h"
#include "../../execution/moe/CPUJointHostGraphEndpoint.h"
#include <omp.h>
#include "../../execution/compute_stages/stages/MoERankBatchSparseStages.h"
#include "../../execution/compute_stages/stages/MoESparseDispatchStage.h"
#include "../../execution/compute_stages/stages/MoESparseReturnReduceStage.h"
#include "../../execution/compute_stages/stages/TPLocalReduceOverlap.h"
#include "../../execution/moe/MoERoutedExpertPlacementPlan.h"
#include "../../execution/moe/RoutedExpertOwnerAssignment.h"
#include "../../execution/moe/MoEExpertOwnerMap.h"
#include "../../execution/moe/MoEExpertOverlayRuntimePlan.h"
#include "../../execution/moe/MoEGroupedVerifierHistogramBoundarySet.h"
#include "../../execution/moe/MoEOverlayParticipantResidency.h"
#include "../../execution/moe/MoEOverlayResidencyAuthority.h"
#include "../../execution/moe/MoEOverlayNodeLocalRouteExchange.h"
#include "../../execution/moe/MoEOverlayNodeLocalRankBatchTransport.h"
#include "../../execution/moe/MoEOverlayRankBatchTransport.h"
#include "../../execution/moe/MoEOverlayRankBatchGraphSchedule.h"
#include "../../execution/moe/MoEOverlaySparseCollective.h"
#include "../../transfer/TransferEngine.h"
#include "../../execution/moe/MoERebalanceController.h"
#include "../../execution/moe/DeviceMoETransferSlotDirectory.h"
#include "../../execution/prefix_cache/PrefixCacheFingerprint.h"
#include "../../backends/BackendManager.h"
#include "../../loaders/GPUVramPreflight.h"
#include "../../memory/BufferId.h"
#include "../../execution/local_execution/graph/GraphResolver.h"
#include "../../planning/ActivationBufferSizing.h"
#include "../../tensors/NativeQuantisedFormatInfo.h"
#include "../../tensors/Tensors.h"
#include "../../utils/DebugEnv.h"
#include "../../utils/PerfStatsCollector.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace llaminar2
{

    namespace
    {
        constexpr char kMoEPrefixRuntimeMagic[8] = {'L', 'M', 'O', 'E', 'R', 'U', 'N', '1'};
        /*
         * Version 7 preserves the exact global whole/cohort route target;
         * a cohort must never be reconstructed from its absent scalar owner.
         * Version 6 binds each portable row to its global model layer, so equal
         * stage shapes cannot restore another stage's request state.
         * Version 5 excludes epoch-ticketed durable placement entirely. Expert
         * residency is model-lifetime state owned by the live RCU authority,
         * whereas a prefix block is request state. Version 4 could snapshot a
         * main table and each MTP sidecar independently, then restore their
         * embedded active_bank fields without updating the shared device epoch
         * selector. Rejecting it prevents a cache hit from creating two
         * placement authorities or rewinding model-wide residency.
         */
        constexpr uint32_t kMoEPrefixRuntimeVersion = 7;

        /**
         * @brief Return whether graph-captured mirrored-layer diagnostics are armed.
         *
         * The switch is intentionally opt-in because every selected boundary
         * adds one small row-copy kernel to the graph. It is read during graph
         * construction, before any capture, so replay topology never changes
         * after an executable has been instantiated.
         */
        bool mirroredLayerDiagnosticsEnabled()
        {
            return DebugEnv::isTruthyEnv(
                "LLAMINAR_MTP_MIRROR_LAYER_DIAGNOSTICS");
        }

        /**
         * @brief Resolve the one GDN layer whose fixed prefill rows are retained.
         *
         * Terminal-row checkpoints are useful for finding the first divergent
         * layer, but a recurrent operator consumes every preceding row. Once a
         * layer has been selected, retaining a logarithmic row sample at each
         * boundary distinguishes an upstream row mismatch from a state-lifetime
         * defect without copying a full activation tensor to the host.
         *
         * The environment value is read only while graph topology is built.
         * Invalid and absent values disable fixed-row sampling; they never
         * silently select another layer.
         */
        std::optional<int> mirroredGDNDiagnosticLayer()
        {
            const char *value =
                DebugEnv::envValue(
                    "LLAMINAR_MTP_GRAPH_REUSE_DIAGNOSTICS_GDN_LAYER");
            if (!value || value[0] == '\0')
                return std::nullopt;

            char *end = nullptr;
            const long parsed = std::strtol(value, &end, 10);
            if (!end || *end != '\0' ||
                parsed < 0 ||
                parsed > std::numeric_limits<int>::max())
            {
                return std::nullopt;
            }
            return static_cast<int>(parsed);
        }

        void appendU32(std::vector<uint8_t> &out, uint32_t value)
        {
            for (int i = 0; i < 4; ++i)
                out.push_back(static_cast<uint8_t>((value >> (8 * i)) & 0xffu));
        }

        void appendI32(std::vector<uint8_t> &out, int32_t value)
        {
            appendU32(out, static_cast<uint32_t>(value));
        }

        void appendU64(std::vector<uint8_t> &out, uint64_t value)
        {
            for (int i = 0; i < 8; ++i)
                out.push_back(static_cast<uint8_t>((value >> (8 * i)) & 0xffu));
        }

        void appendBytes(std::vector<uint8_t> &out, const void *data, size_t bytes)
        {
            const auto *begin = static_cast<const uint8_t *>(data);
            out.insert(out.end(), begin, begin + bytes);
        }

        void appendString(std::vector<uint8_t> &out, const std::string &value)
        {
            appendU32(out, static_cast<uint32_t>(value.size()));
            appendBytes(out, value.data(), value.size());
        }

        /** @brief Decode one little-endian word from an owned archive byte view. */
        bool readU32(std::span<const uint8_t> in, size_t &offset, uint32_t &value)
        {
            if (offset + 4 > in.size())
                return false;
            value = 0;
            for (int i = 0; i < 4; ++i)
                value |= static_cast<uint32_t>(in[offset++]) << (8 * i);
            return true;
        }

        /** @brief Decode a signed word without requiring vector-backed storage. */
        bool readI32(std::span<const uint8_t> in, size_t &offset, int32_t &value)
        {
            uint32_t raw = 0;
            if (!readU32(in, offset, raw))
                return false;
            value = static_cast<int32_t>(raw);
            return true;
        }

        /** @brief Decode a bounded little-endian histogram counter. */
        bool readU64(std::span<const uint8_t> in, size_t &offset, uint64_t &value)
        {
            if (offset + 8 > in.size())
                return false;
            value = 0;
            for (int i = 0; i < 8; ++i)
                value |= static_cast<uint64_t>(in[offset++]) << (8 * i);
            return true;
        }

        /** @brief Decode a length-prefixed runtime-table identity. */
        bool readString(std::span<const uint8_t> in, size_t &offset, std::string &value)
        {
            uint32_t size = 0;
            if (!readU32(in, offset, size) || offset + size > in.size())
                return false;
            value.assign(reinterpret_cast<const char *>(in.data() + offset), size);
            offset += size;
            return true;
        }

        const RoutedExpertLayerPlacement *findExpertOverlayPlacement(
            const MoERoutedExpertPlacementPlan &plan,
            int layer_idx)
        {
            auto it = std::find_if(plan.placements.begin(), plan.placements.end(),
                                   [layer_idx](const RoutedExpertLayerPlacement &placement)
                                   {
                                       return placement.layer == layer_idx;
                                   });
            return it == plan.placements.end() ? nullptr : &(*it);
        }

        bool isUsableExpertOverlayPlacement(
            const RoutedExpertLayerPlacement &placement,
            const MoERoutedExpertPlacementPlan &plan,
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
            const RoutedExpertLayerPlacement &placement,
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

        std::string nodeSuffixForTier(const RoutedExpertTier &tier, int tier_index)
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
            if (!config.moe.routed_expert_plan)
                return nullptr;
            return resolveMoEExpertOverlayRuntimePlan(config.moe.routed_expert_plan);
        }

        /**
         * @brief Return the immutable flattened-row envelope for graph-stable MoE state.
         *
         * Production construction receives @ref GraphConfig::max_activation_rows
         * from the memory-plan admission result.  Direct graph-builder callers
         * have no runner-level admission object, so zero retains their
         * conservative max-sequence envelope.  A negative value is malformed
         * configuration, not a request to silently allocate an arbitrary size.
         */
        int graphStableActivationRowCapacity(
            const GraphConfig &config,
            DeviceId device)
        {
            if (config.max_activation_rows < 0)
            {
                throw std::invalid_argument(
                    "Qwen35 MoE graph activation-row capacity cannot be negative");
            }
            if (config.max_activation_rows > 0)
                return config.max_activation_rows;
            return std::max(
                1,
                resolveActivationBufferSeqLen(config.max_seq_len, device));
        }

        /**
         * @brief Resolve the bounded live-row envelope of an ExpertOverlay segment.
         *
         * Dense activations and KV state retain their independently planned
         * context capacities.  Sparse protocol packets and participant-local
         * captured GEMM families need only hold one root-published segment;
         * using the dense envelope here recreates a full-context H2D/D2H copy
         * for a handful of live routes.
         */
        int overlayPrefillSegmentRowCapacity(
            const GraphConfig &config,
            DeviceId device)
        {
            const int graph_rows =
                graphStableActivationRowCapacity(config, device);
            const bool overlay =
                config.moe.expert_overlay_runtime_plan ||
                (config.moe.routed_expert_plan &&
                 config.moe.routed_expert_plan
                     ->usesExpertOverlayAuthority());
            if (!overlay)
                return graph_rows;

            const int segment_rows =
                config.moe.routed_prefill_config.overlay_segment_rows;
            if (segment_rows <= 0)
            {
                throw std::invalid_argument(
                    "Qwen35 MoE ExpertOverlay prefill segment rows must be positive");
            }
            return std::min(graph_rows, segment_rows);
        }

        /**
         * @brief Return the largest live row count used by any sparse overlay role.
         *
         * MTP grouped verification shares the same serial protocol workspaces
         * and may be wider than an explicitly tiny prefill segment. Keeping
         * that role in the immutable setup envelope avoids a request-time
         * resize while leaving the ordinary prefill scheduler independently
         * tunable.
         */
        int overlaySparseProtocolRowCapacity(
            const GraphConfig &config,
            DeviceId device)
        {
            const int prefill_rows =
                overlayPrefillSegmentRowCapacity(config, device);
            const int verifier_rows =
                retainsMTPGraphCapacity(config.mtp)
                    ? std::max(
                          1,
                          resolveMTPRetainedTargetQueryRows(config.mtp))
                    : 1;
            return std::max(prefill_rows, verifier_rows);
        }

        /**
         * @brief Clamp one graph variant to the admitted sparse packet envelope.
         *
         * A CPU continuation graph may retain the full dense context shape
         * while ExpertOverlay executes its sparse participant work in bounded
         * prefill segments. The compact local-expert tensors follow the sparse
         * packet, not the dense graph's nominal row count. Decode and smaller
         * prefill buckets still select their exact smaller serial family.
         */
        int overlaySparseGraphRowCapacity(
            const GraphConfig &config,
            DeviceId device,
            int graph_rows)
        {
            if (graph_rows <= 0)
            {
                throw std::invalid_argument(
                    "Qwen35 MoE sparse graph row capacity must be positive");
            }
            return std::min(
                graph_rows,
                overlaySparseProtocolRowCapacity(config, device));
        }

        int continuationRootParticipant(const MoERoutedExpertPlacementPlan &plan)
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

        bool expertMaskEnablesExpert(const std::vector<bool> &expert_mask,
                                     int expert,
                                     int num_experts)
        {
            return expert_mask.empty() ||
                   (expert_mask.size() == static_cast<size_t>(num_experts) &&
                    expert >= 0 &&
                    expert < num_experts &&
                    expert_mask[static_cast<size_t>(expert)]);
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
            if (runtime_table->decodeRuntimePublicationRequired(layer_idx))
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
                    !desc.weightsReady() ||
                    !hasMoEExpertFlag(desc.flags, DeviceMoEExpertFlags::Valid) ||
                    !hasMoEExpertFlag(desc.flags, DeviceMoEExpertFlags::Resident) ||
                    !hasMoEExpertFlag(desc.flags, DeviceMoEExpertFlags::LocalCompute))
                {
                    return false;
                }
            }
            return true;
        }

        bool runtimeTableHasUsableFullyReplicatedDecodeBank(
            IMoERuntimeTable *runtime_table,
            int layer_idx,
            int num_experts,
            int top_k,
            int local_participant,
            int participant_count,
            const std::vector<int> &owner_participants)
        {
            if (!runtime_table ||
                layer_idx < 0 ||
                local_participant < 0 ||
                participant_count <= 0 ||
                local_participant >= participant_count ||
                participant_count > static_cast<int>(kDeviceMoEMaxParticipants) ||
                owner_participants.size() != static_cast<size_t>(num_experts) ||
                runtime_table->decodeRuntimePublicationRequired(layer_idx))
            {
                return false;
            }

            const auto &state = runtime_table->hostLayerState(layer_idx);
            if (state.active_bank > 1 ||
                state.active_epoch == 0 ||
                state.expert_count != static_cast<uint32_t>(num_experts) ||
                state.top_k != static_cast<uint32_t>(top_k) ||
                state.participant_id != static_cast<uint32_t>(local_participant) ||
                state.participant_count != static_cast<uint32_t>(participant_count))
            {
                return false;
            }

            const auto &bank = state.banks[state.active_bank];
            if (bank.epoch != state.active_epoch ||
                bank.expert_count != static_cast<uint32_t>(num_experts))
            {
                return false;
            }

            const uint32_t all_participants_mask =
                (1u << static_cast<uint32_t>(participant_count)) - 1u;
            for (int expert = 0; expert < num_experts; ++expert)
            {
                const int owner = owner_participants[static_cast<size_t>(expert)];
                if (owner < 0 || owner >= participant_count)
                    return false;

                const auto &descriptor = bank.experts[static_cast<size_t>(expert)];
                const auto expected_role = static_cast<uint8_t>(
                    owner == local_participant
                        ? DeviceMoEReplicaRole::Primary
                        : DeviceMoEReplicaRole::Replica);
                if (bank.local_compute_mask[static_cast<size_t>(expert)] != 1u ||
                    bank.replica_role[static_cast<size_t>(expert)] != expected_role ||
                    bank.resident_participant_mask[static_cast<size_t>(expert)] !=
                        all_participants_mask ||
                    descriptor.logical_expert_id != expert ||
                    descriptor.owner_participant != owner ||
                    descriptor.local_slot < 0 ||
                    !descriptor.weightsReady() ||
                    !hasMoEExpertFlag(descriptor.flags, DeviceMoEExpertFlags::Valid) ||
                    !hasMoEExpertFlag(descriptor.flags, DeviceMoEExpertFlags::Resident) ||
                    !hasMoEExpertFlag(descriptor.flags, DeviceMoEExpertFlags::Replicated) ||
                    !hasMoEExpertFlag(descriptor.flags, DeviceMoEExpertFlags::LocalCompute) ||
                    (hasMoEExpertFlag(
                         descriptor.flags,
                         DeviceMoEExpertFlags::PreferredOwner) !=
                     (owner == local_participant)))
                {
                    return false;
                }
            }
            return true;
        }

        enum class FullLocalDecodeRuntimePolicy
        {
            SingleParticipant,
            FullyReplicatedLocalTP,
        };

        bool runtimeTableHasUsableMaskedDecodeBank(
            IMoERuntimeTable *runtime_table,
            int layer_idx,
            int num_experts,
            int top_k,
            const std::vector<bool> &expert_mask,
            int local_participant,
            int participant_count,
            const std::vector<int> &owner_participants = {},
            const std::vector<int> &overlay_route_participants = {})
        {
            if (!runtime_table || layer_idx < 0)
                return false;
            if (runtime_table->decodeRuntimePublicationRequired(layer_idx))
                return false;
            if (!expert_mask.empty() &&
                expert_mask.size() != static_cast<size_t>(num_experts))
                return false;
            if (local_participant < 0 ||
                participant_count <= 0 ||
                local_participant >= participant_count)
            {
                return false;
            }
            if (!owner_participants.empty() &&
                owner_participants.size() != static_cast<size_t>(num_experts))
            {
                return false;
            }
            if (!overlay_route_participants.empty() &&
                overlay_route_participants.size() !=
                    static_cast<size_t>(num_experts))
            {
                return false;
            }

            const auto &state = runtime_table->hostLayerState(layer_idx);
            if (state.active_bank > 1 ||
                state.active_epoch == 0 ||
                state.expert_count != static_cast<uint32_t>(num_experts) ||
                state.top_k != static_cast<uint32_t>(top_k) ||
                state.participant_id != static_cast<uint32_t>(local_participant) ||
                state.participant_count != static_cast<uint32_t>(participant_count))
            {
                return false;
            }

            const auto &bank = state.banks[state.active_bank];
            if (bank.epoch != state.active_epoch ||
                bank.expert_count != static_cast<uint32_t>(num_experts))
            {
                return false;
            }

            for (int expert = 0; expert < num_experts; ++expert)
            {
                if (!overlay_route_participants.empty() &&
                    bank.overlay_route_participant[
                        static_cast<size_t>(expert)] !=
                        overlay_route_participants[
                            static_cast<size_t>(expert)])
                {
                    return false;
                }
                const bool expected_local = expertMaskEnablesExpert(expert_mask, expert, num_experts);
                if (bank.local_compute_mask[static_cast<size_t>(expert)] != (expected_local ? 1u : 0u))
                    return false;
                const int expected_owner =
                    owner_participants.empty()
                        ? (expected_local ? local_participant : -1)
                        : owner_participants[static_cast<size_t>(expert)];
                if (expected_owner < -1)
                    return false;
                if (expected_owner >= 0)
                {
                    if (expected_owner >= participant_count)
                        return false;
                    if ((bank.resident_participant_mask[static_cast<size_t>(expert)] &
                         (1u << static_cast<uint32_t>(expected_owner))) == 0u)
                    {
                        return false;
                    }
                    if (bank.experts[static_cast<size_t>(expert)].owner_participant != expected_owner)
                        return false;
                }
                else
                {
                    /*
                     * A per-domain table deliberately leaves experts from
                     * other overlay tiers external. They have no owner or
                     * resident bit in this domain and must never be marked for
                     * local compute. The sparse overlay branch owns those
                     * routes independently.
                     */
                    if (expected_local ||
                        bank.resident_participant_mask[
                            static_cast<size_t>(expert)] != 0u ||
                        bank.experts[static_cast<size_t>(expert)]
                                .owner_participant != -1)
                    {
                        return false;
                    }
                }
                if (!expected_local)
                    continue;

                const auto &desc = bank.experts[static_cast<size_t>(expert)];
                if (desc.logical_expert_id != expert ||
                    desc.local_slot < 0 ||
                    !desc.weightsReady() ||
                    !hasMoEExpertFlag(desc.flags, DeviceMoEExpertFlags::Valid) ||
                    !hasMoEExpertFlag(desc.flags, DeviceMoEExpertFlags::Resident) ||
                    !hasMoEExpertFlag(desc.flags, DeviceMoEExpertFlags::LocalCompute))
                {
                    return false;
                }
            }
            return true;
        }

        bool runtimeTableHasUsableLiveDynamicDecodeBank(
            IMoERuntimeTable *runtime_table,
            int layer_idx,
            int num_experts,
            int top_k,
            int local_participant,
            int participant_count,
            const std::vector<int> &overlay_route_participants,
            const std::string &context)
        {
            if (!runtime_table || layer_idx < 0)
                return false;
            if (runtime_table->decodeRuntimePublicationRequired(layer_idx))
                return false;
            if (local_participant < 0 ||
                participant_count <= 0 ||
                local_participant >= participant_count ||
                participant_count > static_cast<int>(kDeviceMoEMaxParticipants))
            {
                return false;
            }
            if (!overlay_route_participants.empty() &&
                overlay_route_participants.size() !=
                    static_cast<size_t>(num_experts))
            {
                return false;
            }

            const auto &state = runtime_table->hostLayerState(layer_idx);
            auto reject = [&](const std::string &reason)
            {
                if (state.active_epoch != 0)
                {
                    LOG_ERROR("[Qwen35MoEGraph] " << context
                                                  << ": active dynamic MoE decode bank rejected for layer "
                                                  << layer_idx << " epoch " << state.active_epoch
                                                  << ": " << reason);
                }
                return false;
            };
            if (state.active_bank > 1 ||
                state.active_epoch == 0 ||
                state.expert_count != static_cast<uint32_t>(num_experts) ||
                state.top_k != static_cast<uint32_t>(top_k) ||
                state.participant_id != static_cast<uint32_t>(local_participant) ||
                state.participant_count != static_cast<uint32_t>(participant_count))
            {
                std::ostringstream detail;
                detail << "runtime-table metadata does not match decode graph metadata"
                       << " table_active_bank=" << state.active_bank
                       << " table_epoch=" << state.active_epoch
                       << " table_experts=" << state.expert_count
                       << " expected_experts=" << num_experts
                       << " table_top_k=" << state.top_k
                       << " expected_top_k=" << top_k
                       << " table_participant=" << state.participant_id
                       << " expected_participant=" << local_participant
                       << " table_participants=" << state.participant_count
                       << " expected_participants=" << participant_count;
                return reject(detail.str());
            }

            const auto &bank = state.banks[state.active_bank];
            if (bank.epoch != state.active_epoch ||
                bank.expert_count != static_cast<uint32_t>(num_experts))
            {
                return reject("active placement bank metadata does not match runtime-table metadata");
            }

            const uint32_t valid_participant_mask =
                (1u << static_cast<uint32_t>(participant_count)) - 1u;
            const uint32_t local_participant_bit =
                1u << static_cast<uint32_t>(local_participant);
            bool has_local_expert = false;

            for (int expert = 0; expert < num_experts; ++expert)
            {
                if (!overlay_route_participants.empty() &&
                    bank.overlay_route_participant[
                        static_cast<size_t>(expert)] !=
                        overlay_route_participants[
                            static_cast<size_t>(expert)])
                {
                    return reject(
                        "expert " + std::to_string(expert) +
                        " overlay packet target does not match the current global owner map");
                }
                const auto &desc = bank.experts[static_cast<size_t>(expert)];
                const uint32_t raw_resident_mask =
                    bank.resident_participant_mask[static_cast<size_t>(expert)];
                uint32_t effective_resident_mask = raw_resident_mask & valid_participant_mask;
                const bool local_compute =
                    bank.local_compute_mask[static_cast<size_t>(expert)] != 0u;
                const bool external_to_domain =
                    desc.owner_participant == -1 &&
                    effective_resident_mask == 0u &&
                    !local_compute;

                if (desc.owner_participant >= 0 &&
                    desc.owner_participant < participant_count)
                {
                    effective_resident_mask |=
                        1u << static_cast<uint32_t>(desc.owner_participant);
                }

                if ((raw_resident_mask & ~valid_participant_mask) != 0u)
                {
                    return reject("expert " + std::to_string(expert) +
                                  " resident mask names a participant outside the domain");
                }
                if (effective_resident_mask == 0u &&
                    !external_to_domain)
                {
                    return reject("expert " + std::to_string(expert) +
                                  " has no effective resident participant");
                }
                if (desc.logical_expert_id != expert)
                {
                    return reject("expert " + std::to_string(expert) +
                                  " descriptor logical id does not match its table slot");
                }
                if (desc.owner_participant < -1)
                {
                    return reject("expert " + std::to_string(expert) +
                                  " owner participant is malformed");
                }
                if (desc.owner_participant >= participant_count)
                {
                    return reject("expert " + std::to_string(expert) +
                                  " owner participant is outside the domain");
                }

                if (!local_compute)
                    continue;

                if ((effective_resident_mask & local_participant_bit) == 0u)
                {
                    return reject("expert " + std::to_string(expert) +
                                  " is marked local-compute but is not locally resident");
                }
                if (desc.local_slot < 0 ||
                    !desc.weightsReady() ||
                    !hasMoEExpertFlag(desc.flags, DeviceMoEExpertFlags::Valid) ||
                    !hasMoEExpertFlag(desc.flags, DeviceMoEExpertFlags::Resident) ||
                    !hasMoEExpertFlag(desc.flags, DeviceMoEExpertFlags::LocalCompute))
                {
                    return reject("expert " + std::to_string(expert) +
                                  " is marked local-compute without ready packed-weight descriptors");
                }
                has_local_expert = true;
            }
            if (!has_local_expert)
                return reject("placement bank has no local experts");
            return true;
        }

        bool initializeMaskedLocalDecodeRuntimeTable(
            IMoERuntimeTable *runtime_table,
            int layer_idx,
            int num_experts,
            int top_k,
            int d_model,
            int expert_intermediate,
            const std::vector<bool> &expert_mask,
            int local_participant,
            int participant_count,
            const std::vector<int> &owner_participants,
            const std::vector<int> &overlay_route_participants,
            const std::vector<ITensorGemm *> &gate_gemms,
            const std::vector<ITensorGemm *> &up_gemms,
            const std::vector<ITensorGemm *> &down_gemms,
            void *stream,
            bool allow_existing_dynamic_bank,
            const std::string &context)
        {
            if (!runtime_table || layer_idx < 0)
                return false;
            if (auto *device_table =
                    dynamic_cast<DeviceMoERuntimeTable *>(runtime_table);
                device_table && device_table->overlayPlacementSource())
            {
                /*
                 * A child table owns request-local routing and transfer state,
                 * but its initial durable descriptors belong to the canonical
                 * parent. Initialize that parent first, then retain the same
                 * baseline in the child so a later transient apply has a
                 * complete private destination bank. The one-level source
                 * contract is enforced by DeviceMoERuntimeTable.
                 */
                if (!initializeMaskedLocalDecodeRuntimeTable(
                        device_table->overlayPlacementSource(),
                        layer_idx,
                        num_experts,
                        top_k,
                        d_model,
                        expert_intermediate,
                        expert_mask,
                        local_participant,
                        participant_count,
                        owner_participants,
                        overlay_route_participants,
                        gate_gemms,
                        up_gemms,
                        down_gemms,
                        stream,
                        allow_existing_dynamic_bank,
                        context + " canonical parent"))
                {
                    return false;
                }
            }
            if (!expert_mask.empty() &&
                expert_mask.size() != static_cast<size_t>(num_experts))
            {
                LOG_ERROR("[Qwen35MoEGraph] " << context
                                              << ": expert mask size does not match num_experts="
                                              << num_experts);
                return false;
            }
            if (participant_count <= 0 ||
                participant_count > static_cast<int>(kDeviceMoEMaxParticipants) ||
                local_participant < 0 ||
                local_participant >= participant_count)
            {
                LOG_ERROR("[Qwen35MoEGraph] " << context
                                              << ": invalid participant metadata id="
                                              << local_participant
                                              << " count=" << participant_count);
                return false;
            }
            if (!owner_participants.empty() &&
                owner_participants.size() != static_cast<size_t>(num_experts))
            {
                LOG_ERROR("[Qwen35MoEGraph] " << context
                                              << ": owner participant vector size does not match num_experts="
                                              << num_experts);
                return false;
            }
            if (!overlay_route_participants.empty() &&
                overlay_route_participants.size() !=
                    static_cast<size_t>(num_experts))
            {
                LOG_ERROR("[Qwen35MoEGraph] " << context
                                              << ": overlay route participant vector size does not match num_experts="
                                              << num_experts);
                return false;
            }
            for (std::size_t expert = 0; expert < overlay_route_participants.size(); ++expert)
            {
                if (MoEOverlayRouteTarget::fromWord(overlay_route_participants[expert])
                        .isCohort() &&
                    owner_participants[expert] >= 0)
                {
                    LOG_ERROR("[Qwen35MoEGraph] "
                              << context
                              << ": CPU cohort target conflicts with a local GPU-domain owner");
                    return false;
                }
            }

            if (runtimeTableHasUsableMaskedDecodeBank(
                    runtime_table,
                    layer_idx,
                    num_experts,
                    top_k,
                    expert_mask,
                    local_participant,
                    participant_count,
                    owner_participants,
                    overlay_route_participants))
            {
                return true;
            }
            if (allow_existing_dynamic_bank &&
                runtimeTableHasUsableLiveDynamicDecodeBank(
                    runtime_table,
                    layer_idx,
                    num_experts,
                    top_k,
                    local_participant,
                    participant_count,
                    overlay_route_participants,
                    context))
            {
                return true;
            }

            const bool publication_required =
                runtime_table->decodeRuntimePublicationRequired(layer_idx);
            const auto &state = runtime_table->hostLayerState(layer_idx);
            if (!publication_required && state.active_epoch != 0)
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
            update.participant_id = static_cast<uint32_t>(local_participant);
            update.participant_count = static_cast<uint32_t>(participant_count);
            update.experts.resize(static_cast<size_t>(num_experts));
            update.local_compute_mask.assign(static_cast<size_t>(num_experts), 0u);
            update.replica_role.assign(static_cast<size_t>(num_experts),
                                       static_cast<uint8_t>(DeviceMoEReplicaRole::None));
            update.resident_participant_mask.assign(static_cast<size_t>(num_experts), 0u);
            update.overlay_route_participant.assign(
                overlay_route_participants.begin(),
                overlay_route_participants.end());

            bool has_local_expert = false;
            for (int expert = 0; expert < num_experts; ++expert)
            {
                const int expert_owner =
                    owner_participants.empty()
                        ? (expertMaskEnablesExpert(expert_mask, expert, num_experts) ? local_participant : -1)
                        : owner_participants[static_cast<size_t>(expert)];
                if (expert_owner < -1)
                {
                    LOG_ERROR("[Qwen35MoEGraph] " << context
                                                  << ": malformed owner participant for expert "
                                                  << expert << " layer " << layer_idx);
                    return false;
                }
                if (expert_owner >= participant_count)
                {
                    LOG_ERROR("[Qwen35MoEGraph] " << context
                                                  << ": owner participant " << expert_owner
                                                  << " for expert " << expert
                                                  << " is outside participant_count="
                                                  << participant_count);
                    return false;
                }
                if (expert_owner >= 0)
                    update.resident_participant_mask[static_cast<size_t>(expert)] =
                        1u << static_cast<uint32_t>(expert_owner);

                if (!expertMaskEnablesExpert(expert_mask, expert, num_experts))
                {
                    if (expert_owner < 0)
                    {
                        /*
                         * This logical expert belongs to another overlay
                         * domain. Keep its descriptor address-stable but empty;
                         * routing will publish no local row for it and the
                         * explicit sparse branch remains its sole executor.
                         */
                        update.resident_participant_mask[
                            static_cast<size_t>(expert)] = 0u;
                    }
                    update.experts[static_cast<size_t>(expert)].logical_expert_id = expert;
                    update.experts[static_cast<size_t>(expert)].owner_participant = expert_owner;
                    continue;
                }

                if (expert_owner < 0)
                {
                    LOG_ERROR("[Qwen35MoEGraph] " << context
                                                  << ": locally enabled expert "
                                                  << expert
                                                  << " has no owner in the runtime-table domain");
                    return false;
                }

                auto *gate = gate_gemms[static_cast<size_t>(expert)];
                auto *up = up_gemms[static_cast<size_t>(expert)];
                auto *down = down_gemms[static_cast<size_t>(expert)];
                if (!gate || !up || !down)
                {
                    LOG_ERROR("[Qwen35MoEGraph] " << context
                                                  << ": missing prepared GEMM engine for local expert "
                                                  << expert << " layer " << layer_idx);
                    return false;
                }

                DeviceMoEExpertDescriptor desc;
                if (!exportDeviceMoEExpertWeightDescriptors(
                        gate,
                        up,
                        down,
                        d_model,
                        expert_intermediate,
                        desc))
                {
                    LOG_ERROR("[Qwen35MoEGraph] " << context
                                                  << ": prepared GEMM engines for local expert "
                                                  << expert << " layer " << layer_idx
                                                  << " cannot export one uniform NativeQuantised/FP16/BF16/FP32 descriptor family");
                    return false;
                }

                desc.logical_expert_id = expert;
                desc.owner_participant = expert_owner >= 0 ? expert_owner : local_participant;
                desc.local_slot = expert;
                DeviceMoEExpertFlags flags = DeviceMoEExpertFlags::Valid |
                                             DeviceMoEExpertFlags::Resident |
                                             DeviceMoEExpertFlags::LocalCompute;
                if (desc.owner_participant == local_participant)
                    flags |= DeviceMoEExpertFlags::PreferredOwner;
                desc.flags = toMoEExpertFlags(flags);
                update.experts[static_cast<size_t>(expert)] = desc;
                update.local_compute_mask[static_cast<size_t>(expert)] = 1u;
                update.replica_role[static_cast<size_t>(expert)] =
                    static_cast<uint8_t>(DeviceMoEReplicaRole::Primary);
                update.resident_participant_mask[static_cast<size_t>(expert)] |=
                    1u << static_cast<uint32_t>(local_participant);
                has_local_expert = true;
            }

            if (!has_local_expert)
            {
                LOG_ERROR("[Qwen35MoEGraph] " << context
                                              << ": expert mask contains no local experts for layer "
                                              << layer_idx);
                return false;
            }

            try
            {
                runtime_table->prepareInactiveBank(layer_idx, update);
                runtime_table->flipActiveBank(layer_idx, update.epoch, stream);
            }
            catch (const std::exception &ex)
            {
                LOG_ERROR("[Qwen35MoEGraph] " << context
                                              << ": failed to publish masked MoE decode runtime bank for layer "
                                              << layer_idx << ": " << ex.what());
                return false;
            }

            return runtimeTableHasUsableMaskedDecodeBank(
                runtime_table,
                layer_idx,
                num_experts,
                top_k,
                expert_mask,
                local_participant,
                participant_count,
                owner_participants,
                overlay_route_participants);
        }

        bool initializeFullLocalDecodeRuntimeTable(
            IMoERuntimeTable *runtime_table,
            int layer_idx,
            int num_experts,
            int top_k,
            int d_model,
            int expert_intermediate,
            FullLocalDecodeRuntimePolicy topology_policy,
            int local_participant,
            int participant_count,
            const std::vector<int> &owner_participants,
            const std::vector<ITensorGemm *> &gate_gemms,
            const std::vector<ITensorGemm *> &up_gemms,
            const std::vector<ITensorGemm *> &down_gemms,
            void *stream,
            const std::string &context)
        {
            if (!runtime_table || layer_idx < 0)
                return false;
            if (auto *device_table =
                    dynamic_cast<DeviceMoERuntimeTable *>(runtime_table);
                device_table && device_table->overlayPlacementSource())
            {
                if (!initializeFullLocalDecodeRuntimeTable(
                        device_table->overlayPlacementSource(),
                        layer_idx,
                        num_experts,
                        top_k,
                        d_model,
                        expert_intermediate,
                        topology_policy,
                        local_participant,
                        participant_count,
                        owner_participants,
                        gate_gemms,
                        up_gemms,
                        down_gemms,
                        stream,
                        context + " canonical parent"))
                {
                    return false;
                }
            }

            const bool fully_replicated_local_tp =
                topology_policy ==
                FullLocalDecodeRuntimePolicy::FullyReplicatedLocalTP;
            const bool existing_bank_usable =
                fully_replicated_local_tp
                    ? runtimeTableHasUsableFullyReplicatedDecodeBank(
                          runtime_table,
                          layer_idx,
                          num_experts,
                          top_k,
                          local_participant,
                          participant_count,
                          owner_participants)
                    : runtimeTableHasUsableDecodeBank(
                          runtime_table,
                          layer_idx,
                          num_experts,
                          top_k,
                          /*require_full_local_descriptors=*/true);
            if (existing_bank_usable)
            {
                return true;
            }

            const bool publication_required =
                runtime_table->decodeRuntimePublicationRequired(layer_idx);
            const auto &state = runtime_table->hostLayerState(layer_idx);
            if (!publication_required && state.active_epoch != 0)
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
            if (!fully_replicated_local_tp)
            {
                update.local_compute_mask.assign(
                    static_cast<size_t>(num_experts), 1u);
                update.replica_role.assign(
                    static_cast<size_t>(num_experts),
                    static_cast<uint8_t>(DeviceMoEReplicaRole::Primary));
            }

            DeviceMoEExpertFlags descriptor_flags =
                DeviceMoEExpertFlags::Valid |
                DeviceMoEExpertFlags::Resident |
                DeviceMoEExpertFlags::LocalCompute;
            if (!fully_replicated_local_tp)
                descriptor_flags |= DeviceMoEExpertFlags::PreferredOwner;
            const uint32_t flags = toMoEExpertFlags(descriptor_flags);

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
                if (!exportDeviceMoEExpertWeightDescriptors(
                        gate,
                        up,
                        down,
                        d_model,
                        expert_intermediate,
                        desc))
                {
                    LOG_ERROR("[Qwen35MoEGraph] " << context
                                                  << ": prepared GEMM engines for expert "
                                                  << expert << " layer " << layer_idx
                                                  << " cannot export one uniform NativeQuantised/FP16/BF16/FP32 descriptor family");
                    return false;
                }

                desc.logical_expert_id = expert;
                desc.owner_participant = fully_replicated_local_tp ? -1 : 0;
                desc.local_slot = expert;
                desc.flags = flags;
                update.experts[static_cast<size_t>(expert)] = desc;
            }

            if (fully_replicated_local_tp)
            {
                try
                {
                    declareFullyReplicatedPlacementTopology(
                        update,
                        local_participant,
                        participant_count,
                        owner_participants);
                }
                catch (const std::exception &ex)
                {
                    LOG_ERROR("[Qwen35MoEGraph] " << context
                                                  << ": invalid replicated runtime topology for layer "
                                                  << layer_idx << ": " << ex.what());
                    return false;
                }
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

            return fully_replicated_local_tp
                       ? runtimeTableHasUsableFullyReplicatedDecodeBank(
                             runtime_table,
                             layer_idx,
                             num_experts,
                             top_k,
                             local_participant,
                             participant_count,
                             owner_participants)
                       : runtimeTableHasUsableDecodeBank(
                             runtime_table,
                             layer_idx,
                             num_experts,
                             top_k,
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
            fields.push_back({prefix + ".routed_compute_policy", routedExpertComputePolicyToString(domain.routed_compute_policy)});
            fields.push_back({prefix + ".routed_phase_policy", routedExpertPhasePolicyToString(domain.routed_phase_policy)});
            fields.push_back({prefix + ".routed_decode_assignment_policy", routedExpertAssignmentPolicyToString(domain.routed_decode_assignment_policy)});
            fields.push_back({prefix + ".routed_prefill_assignment_policy", routedExpertAssignmentPolicyToString(domain.routed_prefill_assignment_policy)});
            fields.push_back({prefix + ".owner_rank", domain.owner_rank ? std::to_string(*domain.owner_rank) : "-1"});
            appendAddressVectorFields(fields, prefix + ".participant", domain.participants);
            appendRankVectorFields(fields, prefix + ".rank", domain.ranks);
            appendWeightVectorFields(fields, prefix + ".weight", domain.weights);
        }

        void appendExpertDomainFingerprintFields(
            std::vector<PrefixFingerprintField> &fields,
            const RoutedExpertDomain &domain,
            const std::string &prefix)
        {
            fields.push_back({prefix + ".name", domain.name});
            fields.push_back({prefix + ".scope", executionDomainScopeToString(domain.scope)});
            fields.push_back({prefix + ".backend", collectiveBackendTypeToString(domain.backend)});
            fields.push_back({prefix + ".routed_compute_policy", routedExpertComputePolicyToString(domain.routed_compute_policy)});
            fields.push_back({prefix + ".routed_phase_policy", routedExpertPhasePolicyToString(domain.routed_phase_policy)});
            fields.push_back({prefix + ".routed_decode_assignment_policy", routedExpertAssignmentPolicyToString(domain.routed_decode_assignment_policy)});
            fields.push_back({prefix + ".routed_prefill_assignment_policy", routedExpertAssignmentPolicyToString(domain.routed_prefill_assignment_policy)});
            fields.push_back({prefix + ".owner_rank", std::to_string(domain.owner_rank)});
            appendAddressVectorFields(fields, prefix + ".participant", domain.participants);
            appendRankVectorFields(fields, prefix + ".rank", domain.world_ranks);
            appendWeightVectorFields(fields, prefix + ".weight", domain.weights);
        }

        void appendExpertOverlayPlanFingerprintFields(
            std::vector<PrefixFingerprintField> &fields,
            const MoERoutedExpertPlacementPlan &plan,
            const std::string &scope)
        {
            fields.push_back({scope + ".enabled", boolField(plan.enabled)});
            fields.push_back({scope + ".topology", toString(plan.topology)});
            fields.push_back({scope + ".continuation_domain", plan.continuation_domain});
            fields.push_back({scope + ".base_model_domain", plan.base_model_domain});
            fields.push_back({scope + ".effective_base_model_domain", plan.effectiveBaseModelDomain()});
            fields.push_back({scope + ".shared_expert_domain", plan.shared_expert_domain});
            fields.push_back({scope + ".residency_policy", toString(plan.residency_policy)});
            fields.push_back({scope + ".owner_order",
                              routedExpertOwnerOrderToString(plan.owner_order)});
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
                fields.push_back({
                    prefix + ".resolved_live_experts_per_layer.count",
                    std::to_string(
                        tier.resolved_live_experts_per_layer.size())});
                for (size_t layer = 0;
                     layer < tier.resolved_live_experts_per_layer.size();
                     ++layer)
                {
                    fields.push_back({
                        prefix + ".resolved_live_experts_per_layer." +
                            std::to_string(layer),
                        std::to_string(
                            tier.resolved_live_experts_per_layer[layer])});
                }
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
                fields.push_back({prefix + ".scope", executionDomainScopeToString(domain.scope)});
                fields.push_back({prefix + ".backend", collectiveBackendTypeToString(domain.backend)});
                fields.push_back({prefix + ".routed_compute_policy", routedExpertComputePolicyToString(domain.routed_compute_policy)});
                fields.push_back({prefix + ".routed_phase_policy", routedExpertPhasePolicyToString(domain.routed_phase_policy)});
                fields.push_back({prefix + ".routed_decode_assignment_policy", routedExpertAssignmentPolicyToString(domain.routed_decode_assignment_policy)});
                fields.push_back({prefix + ".routed_prefill_assignment_policy", routedExpertAssignmentPolicyToString(domain.routed_prefill_assignment_policy)});
                fields.push_back({prefix + ".primary_participant", domain.primary_participant.toString()});
                fields.push_back({prefix + ".primary_device", domain.primary_device.to_string()});
                fields.push_back({prefix + ".primary_world_rank", std::to_string(domain.primary_world_rank)});
                fields.push_back({prefix + ".primary_world_rank_known", boolField(domain.primary_world_rank_known)});
                fields.push_back({prefix + ".owner_rank", std::to_string(domain.owner_rank)});
                fields.push_back({prefix + ".primary_is_local", boolField(domain.primary_is_local)});
                fields.push_back({prefix + ".primary_owned_by_current_rank",
                                  boolField(domain.primary_owned_by_current_rank)});
                fields.push_back({prefix + ".local_reachable_for_mvp", boolField(domain.local_reachable_for_mvp)});

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

        const RoutedExpertDomain *expertDomainForTier(
            const MoERoutedExpertPlacementPlan &plan,
            const RoutedExpertTier &tier)
        {
            auto it = std::find_if(plan.domains.begin(), plan.domains.end(),
                                   [&](const RoutedExpertDomain &domain)
                                   {
                                       return domain.name == tier.domain;
                                   });
            return it == plan.domains.end() ? nullptr : &(*it);
        }

        bool isLocalTPExpertIdApportionedTier(
            const MoERoutedExpertPlacementPlan &plan,
            const RoutedExpertTier &tier)
        {
            const auto *domain = expertDomainForTier(plan, tier);
            return domain &&
                   domain->scope == ExecutionDomainScope::RANK_LOCAL &&
                   domain->routed_compute_policy == RoutedExpertComputePolicy::Apportioned &&
                   domain->participants.size() > 1;
        }

        /**
         * @brief Return whether one routed tier fully mirrors experts locally.
         *
         * A replicated LocalTP tier is physically multi-participant but each
         * participant owns every complete expert. Its graph therefore performs
         * the full routed FFN locally and must not reduce duplicate outputs.
         */
        bool isLocalTPReplicatedTier(
            const MoERoutedExpertPlacementPlan &plan,
            const RoutedExpertTier &tier)
        {
            const auto *domain = expertDomainForTier(plan, tier);
            return domain &&
                   domain->scope == ExecutionDomainScope::RANK_LOCAL &&
                   domain->routed_compute_policy ==
                       RoutedExpertComputePolicy::Replicated &&
                   domain->participants.size() > 1;
        }

        /**
         * @brief Return whether replicated residents split only ordinary prefill.
         *
         * The physical compute policy remains replicated so every participant
         * can execute every expert during decode. This phase policy changes only
         * ordinary prefill scheduling; grouped verifier rows remain decode and
         * therefore retain complete local execution.
         */
        bool isPrefillApportionedDecodeReplicatedTier(
            const MoERoutedExpertPlacementPlan &plan,
            const RoutedExpertTier &tier)
        {
            const auto *domain = expertDomainForTier(plan, tier);
            return domain &&
                   domain->routed_compute_policy ==
                       RoutedExpertComputePolicy::Replicated &&
                   domain->routed_phase_policy ==
                       RoutedExpertPhasePolicy::
                           PrefillApportionedDecodeReplicated;
        }

        bool canUseLocalTPExpertIdApportionedFastPath(
            const MoERoutedExpertPlacementPlan &plan,
            const DeviceId &device,
            const RoutedExpertTier **out_tier = nullptr)
        {
            if (!plan.usesExpertOverlayAuthority() ||
                plan.routed_tiers.size() != 1 ||
                plan.continuation_domain != plan.routed_tiers.front().domain)
            {
                return false;
            }

            const RoutedExpertTier &tier = plan.routed_tiers.front();
            if (!isLocalTPExpertIdApportionedTier(plan, tier))
                return false;

            const auto *domain = expertDomainForTier(plan, tier);
            if (!domain)
                return false;

            const bool contains_device = std::any_of(
                domain->participants.begin(),
                domain->participants.end(),
                [&](const GlobalDeviceAddress &address)
                {
                    return address.toLocalDeviceId() == device;
                });
            if (!contains_device)
                return false;

            if (out_tier)
                *out_tier = &tier;
            return true;
        }

        /**
         * @brief Test whether a graph device participates in one replicated tier.
         *
         * The shape intentionally mirrors the apportioned fast-path predicate:
         * one continuation-local tier keeps graph ownership symmetric and lets
         * every device bind its own complete prepared expert registry.
         */
        bool canUseLocalTPReplicatedFastPath(
            const MoERoutedExpertPlacementPlan &plan,
            const DeviceId &device,
            const RoutedExpertTier **out_tier = nullptr)
        {
            if (!plan.usesExpertOverlayAuthority() ||
                plan.routed_tiers.size() != 1 ||
                plan.continuation_domain != plan.routed_tiers.front().domain)
            {
                return false;
            }

            const RoutedExpertTier &tier = plan.routed_tiers.front();
            if (!isLocalTPReplicatedTier(plan, tier))
                return false;

            const auto *domain = expertDomainForTier(plan, tier);
            if (!domain)
                return false;

            const bool contains_device = std::any_of(
                domain->participants.begin(),
                domain->participants.end(),
                [&](const GlobalDeviceAddress &address)
                {
                    return address.toLocalDeviceId() == device;
                });
            if (!contains_device)
                return false;

            if (out_tier)
                *out_tier = &tier;
            return true;
        }

        bool planHasLocalTPApportionedExpertDomain(
            const MoERoutedExpertPlacementPlan &plan)
        {
            for (const auto &tier : plan.routed_tiers)
            {
                if (isLocalTPExpertIdApportionedTier(plan, tier))
                    return true;
            }
            return false;
        }

        bool planHasLocalTPPrefillApportionedDecodeReplicatedDomain(
            const MoERoutedExpertPlacementPlan &plan)
        {
            for (const auto &tier : plan.routed_tiers)
            {
                if (isLocalTPReplicatedTier(plan, tier) &&
                    isPrefillApportionedDecodeReplicatedTier(plan, tier))
                {
                    return true;
                }
            }
            return false;
        }

        std::optional<uint32_t> expectedPrefixRuntimeParticipantCount(
            const GraphConfig &config)
        {
            const auto *local_tp_ctx = dynamic_cast<const ILocalTPContext *>(config.tp_ctx);
            if (!local_tp_ctx || local_tp_ctx->degree() <= 1)
                return std::nullopt;

            const bool expert_id_apportioned =
                config.moe.routed_compute_policy == RoutedExpertComputePolicy::Apportioned ||
                config.moe.routed_compute_policy == RoutedExpertComputePolicy::GateUpOwnedDownColumns ||
                (config.moe.routed_expert_plan &&
                 (planHasLocalTPApportionedExpertDomain(*config.moe.routed_expert_plan) ||
                  planHasLocalTPPrefillApportionedDecodeReplicatedDomain(
                      *config.moe.routed_expert_plan)));
            if (!expert_id_apportioned)
                return std::nullopt;

            return static_cast<uint32_t>(local_tp_ctx->degree());
        }

        bool portableRuntimeLayerHasPrefixRestoreState(
            const DeviceMoEPortableLayerRuntimeState &layer)
        {
            /*
             * Placement epoch alone does not prove request-owned runtime state:
             * portable restore flips every table layer so sparse MTP sidecar
             * tables can contain epoch-only placeholder layers.  Conversely,
             * epoch zero does not mean "empty" either. Dynamic and LLEP
             * expert-overlay prefill can materialize logical owner/local-compute
             * placement before the first active placement-bank epoch advances,
             * and it can also accumulate routing histograms before a bank flip.
             *
             * A prefix snapshot therefore keeps layers with actual logical
             * placement or routing evidence, regardless of epoch, and ignores
             * layers whose only signal is a restored placeholder epoch.
             */
            const auto has_logical_placement =
                [](const DeviceMoEPortableExpertRuntimeState &expert)
            {
                return expert.overlay_route_participant != -1 ||
                       expert.owner_participant >= 0 ||
                       expert.local_slot >= 0 ||
                       expert.flags != 0u ||
                       expert.local_compute != 0u ||
                       expert.replica_role !=
                           static_cast<uint8_t>(DeviceMoEReplicaRole::None) ||
                       expert.resident_participant_mask != 0u;
            };
            const bool has_placement =
                std::any_of(layer.experts.begin(),
                            layer.experts.end(),
                            has_logical_placement);
            if (has_placement)
            {
                return true;
            }

            const auto has_nonzero_count = [](const std::vector<uint64_t> &counts)
            {
                return std::any_of(
                    counts.begin(),
                    counts.end(),
                    [](uint64_t count)
                    { return count != 0u; });
            };
            return has_nonzero_count(layer.selected_histogram) ||
                   has_nonzero_count(layer.local_histogram);
        }

        bool portableRuntimeStateMatchesPrefixRestoreDomain(
            const std::string &table_key,
            const std::vector<DeviceMoEPortableLayerRuntimeState> &layers,
            const std::optional<uint32_t> &expected_participant_count,
            DeviceId device)
        {
            if (layers.empty())
                return false;

            bool has_restore_state = false;
            for (size_t layer_idx = 0; layer_idx < layers.size(); ++layer_idx)
            {
                const auto &layer = layers[layer_idx];
                const bool layer_has_restore_state =
                    portableRuntimeLayerHasPrefixRestoreState(layer);
                has_restore_state = has_restore_state || layer_has_restore_state;
                if (expected_participant_count &&
                    layer_has_restore_state &&
                    layer.participant_count != *expected_participant_count)
                {
                    PerfStatsCollector::addCounter(
                        "prefix_cache",
                        "moe_portable_runtime_state_skipped_incompatible_domain",
                        1.0,
                        "prefix_cache",
                        device.toString(),
                        {{"table", table_key},
                         {"layer", std::to_string(layer.model_layer)},
                         {"snapshot_participants", std::to_string(layer.participant_count)},
                         {"expected_participants", std::to_string(*expected_participant_count)}});
                    LOG_DEBUG("[Qwen35MoEGraph] Skipping prefix-cache MoE runtime state for "
                              << table_key << " layer=" << layer.model_layer
                              << ": snapshot participant_count=" << layer.participant_count
                              << " does not match restore domain participant_count="
                              << *expected_participant_count);
                    return false;
                }
            }

            if (!has_restore_state)
            {
                PerfStatsCollector::addCounter(
                    "prefix_cache",
                    "moe_portable_runtime_state_skipped_empty_request_state",
                    1.0,
                    "prefix_cache",
                    device.toString(),
                    {{"table", table_key}});
            }
            return has_restore_state;
        }

        /**
         * @brief Borrow the existing stage-local captured maintenance authority.
         * @param tp_ctx Original native context admitted by the stage factory.
         * @param device Exact graph participant device.
         * @param tp_device_idx Original participant index in that same context.
         * @return True only when enabled policy and the real native family both qualify.
         */
        bool isHomogeneousGpuLocalTPRebalanceDomain(
            const ILocalTPContext &tp_ctx,
            DeviceId device,
            int tp_device_idx)
        {
            return debugEnv().moe_rebalance.device_rebalance_maintenance_graph &&
                isHomogeneousGPULocalTPMaintenanceDomain(tp_ctx, device, tp_device_idx,
                    static_cast<int>(kDeviceMoEMaxParticipants));
        }

        std::optional<DeviceMoERebalanceTransferMode> selectGraphRebalanceTransferMode(
            const ILocalTPContext &tp_ctx,
            const DeviceId &destination_device,
            uint32_t max_hot_replicas_per_participant,
            bool requires_expert_payload_movement)
        {
            const auto &moe_env = debugEnv().moe_rebalance;
            if (moe_env.device_rebalance_payload_sideband ||
                moe_env.allow_legacy_collective_rebalance_transfer)
            {
                LOG_ERROR("[Qwen35MoEGraph] Device-side MoE rebalance payload migration is disabled on "
                          << destination_device.to_string()
                          << " because fixed-size collective payload arenas move empty expert slots. "
                             "Use CompactTransferSlots async maintenance for non-empty transfer-slot arrivals.");
                return std::nullopt;
            }

            if (max_hot_replicas_per_participant == 0 &&
                !requires_expert_payload_movement)
            {
                LOG_DEBUG("[Qwen35MoEGraph] Device-side MoE rebalance for "
                          << destination_device.to_string()
                          << " is resident-only because no expert payload movement is enabled; "
                             "host publish/apply fallback remains disabled");
                return DeviceMoERebalanceTransferMode::ResidentOnly;
            }

            if (moe_env.device_rebalance_maintenance_graph)
            {
                if (!tp_ctx.supportsRawAllgatherOnStreamGraphCapture())
                {
                    LOG_ERROR("[Qwen35MoEGraph] Async device-side MoE rebalance maintenance on "
                              << destination_device.to_string()
                              << " requires graph-capturable raw NCCL/RCCL allgather on an explicit stream.");
                    return std::nullopt;
                }

                LOG_TRACE("[Qwen35MoEGraph] Device-side MoE rebalance will collect histogram and compact "
                          "arrival metadata "
                          "with the async maintenance graph on "
                          << destination_device.to_string()
                          << "; decode collectives will not carry rebalance histogram sidebands");
                return DeviceMoERebalanceTransferMode::CompactTransferSlots;
            }

            LOG_ERROR("[Qwen35MoEGraph] Device-side MoE rebalance requires "
                         "LLAMINAR_MOE_DEVICE_REBALANCE_MAINTENANCE_GRAPH=1 so histogram "
                         "state moves on the async rolling-wave maintenance lane for "
                      << destination_device.to_string()
                      << ". Host publish/apply fallback is refused.");
            return std::nullopt;
        }

        int gpuOrdinalForGraphDevice(DeviceId device)
        {
            if (device.is_cuda())
                return device.cuda_ordinal();
            if (device.is_rocm())
                return device.rocm_ordinal();
            return -1;
        }

        /**
         * @brief Derive one expert's transfer-slot format from compute-ready GEMMs.
         *
         * The ExpertGemmRegistry is the authoritative source consumed by GPU
         * execution. Reading its exported descriptors keeps transfer storage
         * sizing tied to the exact codebook and packed representation that the
         * grouped kernels will execute. Raw model tensors are intentionally not
         * consulted here: doing so creates two format authorities and fails for
         * legitimate prepacked or test-provided engines.
         *
         * @param gate Prepared gate projection for one logical expert.
         * @param up Prepared up projection for the same logical expert.
         * @param down Prepared down projection for the same logical expert.
         * @param d_model Declarative hidden dimension.
         * @param expert_intermediate Declarative expert intermediate dimension.
         * @return Three exact projection specs, or std::nullopt when any engine
         *         cannot export a valid, geometry-compatible weight descriptor.
         */
        std::optional<std::vector<DeviceMoETransferSlotDirectory::ProjectionSpec>>
        transferSlotSpecsFromPreparedExpertEngines(
            ITensorGemm *gate,
            ITensorGemm *up,
            ITensorGemm *down,
            int d_model,
            int expert_intermediate)
        {
            if (!gate || !up || !down ||
                d_model <= 0 || expert_intermediate <= 0)
            {
                return std::nullopt;
            }

            auto make_spec =
                [](const char *label, ITensorGemm *engine, int n, int k)
                -> std::optional<DeviceMoETransferSlotDirectory::ProjectionSpec>
            {
                if (!engine || n <= 0 || k <= 0)
                    return std::nullopt;

                ContiguousFloatingPointWeightDescriptor floating{};
                if (engine->exportContiguousFloatingPointWeights(floating))
                {
                    DeviceMoEFloatingMatrixDesc matrix;
                    DeviceMoEWeightFormat format;
                    if (!exportDeviceMoEFloatingMatrixDescriptor(floating, matrix, format) ||
                        matrix.n != n || matrix.k != k)
                        return std::nullopt;
                    return DeviceMoETransferSlotDirectory::ProjectionSpec{
                        .label = label, .N = n, .K = k,
                        .format = ExpertWeightFormat::floating(floating.type),
                    };
                }

                DeviceNativeQuantisedMatrixDesc descriptor{};
                NativeQuantisedSourceIdentity source_identity{};
                if (!engine->exportNativeQuantisedMatrixDesc(descriptor) ||
                    !engine->exportNativeQuantisedSourceIdentity(source_identity) ||
                    !source_identity.present ||
                    native_quantised_formats::forSourceIdentity(
                        source_identity.codebook_id,
                        source_identity.is_superblock) == nullptr ||
                    !descriptor.valid() ||
                    descriptor.n != n ||
                    descriptor.k != k)
                {
                    return std::nullopt;
                }
                if (descriptor.source_identity_present &&
                    (descriptor.source_codebook_id !=
                         source_identity.codebook_id ||
                     static_cast<bool>(descriptor.source_is_superblock) !=
                         source_identity.is_superblock))
                {
                    return std::nullopt;
                }

                uint8_t payload_bytes_per_block = 0;
                uint8_t has_mins = 0;
                uint8_t has_emins = 0;
                if (!deviceMoEProjectionAllocationCapacity(
                        descriptor,
                        payload_bytes_per_block,
                        has_mins,
                        has_emins))
                {
                    return std::nullopt;
                }

                DeviceMoETransferSlotDirectory::ProjectionSpec spec;
                spec.label = label;
                spec.N = n;
                spec.K = k;
                spec.payload_bytes_per_block =
                    static_cast<int>(payload_bytes_per_block);
                spec.is_asymmetric = has_mins != 0;
                spec.has_emins = has_emins != 0;
                spec.codebook_id = descriptor.codebook_id;
                spec.format =
                    ExpertWeightFormat::nativeQuantised(source_identity);
                return spec;
            };

            auto gate_spec = make_spec(
                "gate",
                gate,
                expert_intermediate,
                d_model);
            auto up_spec = make_spec(
                "up",
                up,
                expert_intermediate,
                d_model);
            auto down_spec = make_spec(
                "down",
                down,
                d_model,
                expert_intermediate);
            if (!gate_spec || !up_spec || !down_spec)
                return std::nullopt;

            std::vector<DeviceMoETransferSlotDirectory::ProjectionSpec> specs;
            specs.reserve(3);
            specs.push_back(std::move(*gate_spec));
            specs.push_back(std::move(*up_spec));
            specs.push_back(std::move(*down_spec));
            return specs;
        }

        /**
         * @brief Resolve the one graph-local logical participant in a routed tier.
         * @param owner_map Original immutable expert-owner authority.
         * @param tier_index Exact routed tier, independent of dense TP ordering.
         * @param device Physical storage/backend identity of this graph.
         * @param graph_scope Source-bound endpoint selection for graphs whose
         *        logical participants can share one coarse DeviceId. An absent
         *        view is valid only when the physical member is unique.
         * @return Global owner-map participant id, or -1 when this graph owns no member.
         * @throws std::invalid_argument For a foreign view or ambiguous ownership.
         *
         * Domain participant indices and global owner-map ids are different
         * namespaces. In particular both CPU NUMA endpoints use DeviceId::cpu();
         * the existing LocalTP endpoint authenticates their domain/rank/index
         * tuple before mapping it into the global routing/residency namespace.
         */
        int participantIdForTierDevice(
            const MoEExpertOwnerMap &owner_map,
            int tier_index,
            DeviceId device,
            const MoEExpertOverlayGraphScope *graph_scope = nullptr)
        {
            if (graph_scope && graph_scope->device() != device)
                throw std::invalid_argument("ExpertOverlay graph lookup has a foreign storage identity");
            int selected = -1;
            for (const auto &participant : owner_map.participants())
            {
                if (participant.tier_idx != tier_index || !participant.device.is_valid() || participant.device != device)
                    continue;
                if (graph_scope && (!participant.world_rank_known ||
                    !graph_scope->contains(participant.domain_name, participant.device,
                        participant.world_rank, participant.domain_participant_index)))
                    continue;
                if (selected >= 0)
                    throw std::invalid_argument("ExpertOverlay graph lookup maps one endpoint to multiple tier participants");
                selected = participant.participant_id;
            }
            return selected;
        }

        /**
         * @brief Immutable graph-lowering facts for the continuation tier.
         *
         * `graph_local_participant` uses the global overlay identity because
         * sparse dispatch and residency banks use that namespace. Device
         * runtime tables instead use the participant's domain-local index;
         * callers obtain it from the owner-map participant record.
         */
        using RoutedContinuationTopology = QwenRoutedContinuationTopology;

        /**
         * @brief One graph's immutable role in a distributed sparse MoE round trip.
         *
         * A distributed dense continuation has more than one full-model graph,
         * but exactly one of those graphs owns route dispatch.  The other CPU
         * NodeTP graphs are rank-batch targets: they receive the authenticated
         * sparse packet, execute their participant-local experts, return the
         * compact rows, and then join the rooted dense publication.  Captured
         * GPU continuation peers have a distinct role because their local
         * branch participates in the retained device timeline rather than the
         * portable rank-batch host boundary.
         */
        enum class DistributedSparseGraphRole : uint8_t
        {
            LocalAuthority,              ///< One-process graph owns the complete sparse transaction.
            ContinuationSource,          ///< Distributed logical root dispatches and reduces returns.
            RankBatchContinuationTarget, ///< CPU NodeTP follower receives, computes, and returns rows.
            CapturedContinuationPeer,    ///< GPU continuation peer joins the retained device transaction.
        };

        /** @brief Typed predicates derived from one distributed graph role. */
        struct DistributedSparseGraphContract
        {
            DistributedSparseGraphRole role =
                DistributedSparseGraphRole::LocalAuthority;

            /** @return Whether this graph is the sole route-dispatch authority. */
            bool ownsDispatchAuthority() const noexcept
            {
                return role == DistributedSparseGraphRole::LocalAuthority ||
                       role == DistributedSparseGraphRole::ContinuationSource;
            }

            /** @return Whether this graph crosses a portable rank-batch boundary. */
            bool participatesInRankBatchProtocol() const noexcept
            {
                return role == DistributedSparseGraphRole::ContinuationSource ||
                       role ==
                           DistributedSparseGraphRole::RankBatchContinuationTarget;
            }

            /** @return Whether this graph owns the target half of rank batching. */
            bool isRankBatchTarget() const noexcept
            {
                return role ==
                       DistributedSparseGraphRole::RankBatchContinuationTarget;
            }
        };

        /**
         * @brief Resolve the graph role once from frozen topology ownership.
         * @param distributed_overlay Whether more than one MPI rank participates.
         * @param captured_distributed_device Whether this graph is a retained
         *        GPU continuation participant.
         * @param owns_continuation_root Whether this process/device owns the
         *        plan's authenticated logical continuation root.
         * @return One complete role; callers must not reconstruct it from flags.
         */
        DistributedSparseGraphContract resolveDistributedSparseGraphContract(
            bool distributed_overlay,
            bool captured_distributed_device,
            bool owns_continuation_root) noexcept
        {
            if (!distributed_overlay)
            {
                return {
                    .role =
                        DistributedSparseGraphRole::LocalAuthority,
                };
            }
            if (owns_continuation_root)
            {
                return {
                    .role =
                        DistributedSparseGraphRole::ContinuationSource,
                };
            }
            return {
                .role = captured_distributed_device
                            ? DistributedSparseGraphRole::
                                  CapturedContinuationPeer
                            : DistributedSparseGraphRole::
                                  RankBatchContinuationTarget,
            };
        }

        /**
         * @brief Resolve the one continuation tier before runtime-table choice.
         *
         * Decode graphs are commonly constructed before prefill graphs. The
         * capture decision must therefore be derived from declarative topology
         * rather than from whether another graph happened to materialize a
         * table first. A heterogeneous overlay may still contain a homogeneous
         * rank-local continuation domain; that domain executes its local MoE
         * branch inside the complete captured CUDA/HIP graph while the exact
         * logical root crosses the explicit sparse boundary to other ranks.
         *
         * @param plan Canonical routed-expert placement plan.
         * @param device Device whose graph is being lowered.
         * @param local_tp_ctx Rank-local TP collective context, if any.
         * @param local_tp_device_index Device index within @p local_tp_ctx.
         * @param overlay_runtime_available Whether a rank namespace is bound,
         *        including the intentional one-rank heterogeneous case.
         * @param current_world_rank MPI owner of the graph being lowered.
         * @return Typed continuation topology used by every later lowering.
         * @throws std::runtime_error for ambiguous or internally inconsistent
         *         continuation-domain declarations.
         */
        RoutedContinuationTopology resolveRoutedContinuationTopology(
            const MoERoutedExpertPlacementPlan &plan,
            const MoEExpertOwnerMap *prepared_owners,
            DeviceId device,
            ILocalTPContext *local_tp_ctx,
            int local_tp_device_index,
            bool overlay_runtime_available,
            int current_world_rank)
        {
            RoutedContinuationTopology result;
            for (size_t tier_index = 0;
                 tier_index < plan.routed_tiers.size();
                 ++tier_index)
            {
                const auto &tier = plan.routed_tiers[tier_index];
                if (tier.domain != plan.continuation_domain)
                    continue;
                if (result.tier_index >= 0)
                {
                    throw std::runtime_error(
                        "Qwen35 MoE continuation domain names multiple routed tiers");
                }
                result.tier_index = static_cast<int>(tier_index);
                result.domain = expertDomainForTier(plan, tier);
            }

            if (!overlay_runtime_available || !result.domain ||
                !device.is_gpu() || current_world_rank < 0)
            {
                return result;
            }

            if (!prepared_owners)
                throw std::logic_error("Qwen35 MoE continuation has no prepared canonical owner map");
            const auto &owner_map = *prepared_owners;
            for (const auto &candidate : owner_map.participants())
            {
                if (candidate.tier_idx != result.tier_index ||
                    candidate.device != device ||
                    !candidate.world_rank_known ||
                    candidate.world_rank != current_world_rank)
                {
                    continue;
                }
                if (result.graph_local_participant >= 0)
                {
                    throw std::runtime_error(
                        "Qwen35 MoE distributed continuation resolves more than one graph-local participant for " +
                        device.to_string());
                }
                result.graph_local_participant = candidate.participant_id;
            }
            if (result.graph_local_participant < 0)
                return result;

            const auto *participant = owner_map.participantForId(
                result.graph_local_participant);
            if (!participant ||
                participant->domain_name != plan.continuation_domain)
            {
                throw std::runtime_error(
                    "Qwen35 MoE distributed continuation has an unaligned "
                    "graph-device/participant mapping for " +
                    device.to_string());
            }

            const bool single_device_domain =
                result.domain->scope == ExecutionDomainScope::SINGLE &&
                result.domain->participants.size() == 1u &&
                result.domain->routed_compute_policy !=
                    RoutedExpertComputePolicy::TensorSharded;
            /*
             * A rank-local routed-expert domain remains multi-participant even
             * when dense weights are replicated. Dense tensor parallelism and
             * whole-expert apportionment are independent axes: every expert
             * participant still needs its captured local branch and one
             * continuation publication. Coupling this predicate to the dense
             * policy made replicated dense execution silently lose local
             * experts and is therefore structurally invalid.
             */
            const bool rank_local_tp_domain =
                local_tp_ctx && local_tp_ctx->degree() > 1 &&
                result.domain->scope == ExecutionDomainScope::RANK_LOCAL &&
                (result.domain->routed_compute_policy == RoutedExpertComputePolicy::Apportioned ||
                 hasMappedGPUCPUIntermediateShardTopology(plan)) &&
                static_cast<int>(result.domain->participants.size()) ==
                    local_tp_ctx->degree();
            if (!single_device_domain && !rank_local_tp_domain)
            {
                result.graph_local_participant = -1;
                return result;
            }
            if (rank_local_tp_domain &&
                participant->domain_participant_index !=
                    local_tp_device_index)
            {
                throw std::runtime_error(
                    "Qwen35 MoE distributed LocalTP continuation has an "
                    "unaligned graph-device/participant mapping for " +
                    device.to_string());
            }

            result.captured_continuation_device = true;
            result.captured_local_tp = rank_local_tp_domain;
            return result;
        }
    } // namespace
    struct Qwen35MoEGraph::FFNBuildContext
    {
        struct SharedPrefillCPUInputFork
        {
            std::string publication_node;
            std::string routed_expert_node;
        };

        struct CanonicalMoEPublicationLowering
        {
            MoEParticipantPublicationPolicy policy =
                MoEParticipantPublicationPolicy::
                    IndependentBranchCollectives;
            int root_participant = -1;
            int participant_count = 0;
            std::string routed_producer;
            std::string rooted_reduce_node;
            std::string post_collective_terminal;

            [[nodiscard]] bool usesRankBanks() const noexcept
            {
                return policy == MoEParticipantPublicationPolicy::
                                     CanonicalRootedRankBanks;
            }

            [[nodiscard]] bool usesCanonicalRouteSlots() const noexcept
            {
                return policy != MoEParticipantPublicationPolicy::
                                     IndependentBranchCollectives;
            }

            [[nodiscard]] bool usesRootedCollective() const noexcept
            {
                return policy == MoEParticipantPublicationPolicy::
                                     CanonicalRootedRouteSlots ||
                       usesRankBanks();
            }

            [[nodiscard]] bool usesAllreduceRouteSlots() const noexcept
            {
                return policy == MoEParticipantPublicationPolicy::
                                     CanonicalAllreduceRouteSlots;
            }

            [[nodiscard]] bool usesMappedSparseRoutes() const noexcept
            {
                return policy == MoEParticipantPublicationPolicy::
                                     CanonicalMappedRoutesRootedShared;
            }

            [[nodiscard]] bool usesPackedRouteGather() const noexcept
            {
                return policy == MoEParticipantPublicationPolicy::
                                     CanonicalRootedPackedRouteRows;
            }
        };

        struct DeferredOverlayCombinedPublication
        {
            enum class CaptureRendezvous : std::uint8_t
            {
                NativeFullGraph,
                ExplicitWave,
            };

            int root_device_index = -1;
            int participant_count = 0;
            std::string routed_terminal;
            CaptureRendezvous capture_rendezvous =
                CaptureRendezvous::ExplicitWave;
            std::string capture_wave_identity;

            [[nodiscard]] bool valid() const noexcept
            {
                return root_device_index >= 0 && participant_count > 1 &&
                       root_device_index < participant_count &&
                       !routed_terminal.empty() &&
                       ((capture_rendezvous ==
                             CaptureRendezvous::NativeFullGraph &&
                         capture_wave_identity.empty()) ||
                        (capture_rendezvous ==
                             CaptureRendezvous::ExplicitWave &&
                         !capture_wave_identity.empty()));
            }
        };

        const LayerWeights &layer;
        ActivationBuffers &buffers;
        const LayerGraphSourceIdentity &identity;
        int seq_len;
        int batch_size;
        DeviceId device;
        void *device_state_publication_stream;
        const int32_t *sequence_lengths_device;
        const int32_t *absolute_position_ids_device;
        const Qwen4ExpPredictorResidualBinding *predictor;

        int layer_idx;
        ComputeGraph &graph;
        bool mtp_sidecar_context;
        int mtp_depth_idx;
        const std::string &prefix;
        std::string &ffn_terminal;
        int total_tokens;
        const int32_t *routed_live_rows;
        MoEOverlayServicePhaseHint routed_service_phase;
        ExpertHistogramSource routed_histogram_source;
        const std::shared_ptr<const MoEExpertOverlayRuntimePlan> &overlay_runtime_plan;
        const std::shared_ptr<const MoERoutedExpertPlacementPlan> &overlay_plan;
        const RoutedExpertLayerPlacement *overlay_placement;
        bool mapped_intermediate_shards;
        MoEOverlayServicePhaseHint mapped_shard_phase;
        bool mapped_shard_grouped;
        bool mapped_cuda_retained_m1;
        bool mapped_decode_shared_overlap;
        const LayerWeightBindings &layer_bindings;
        IMoERuntimeTable *moe_runtime_table;
        bool ordinary_prefill_graph;
        MoEOverlayNodeLocalRouteTransport node_local_route_transport;
        bool last_device_rebalance_decode_layer;
        ILocalTPContext *local_tp_ctx;
        IGlobalTPContext *global_tp_ctx;
        bool distributed_expert_overlay;
        const QwenRoutedContinuationTopology &routed_continuation_topology;
        bool captured_overlay_continuation;
        bool captured_local_tp_continuation;
        const RoutedExpertDomain *cpu_llep_overlay_domain;
        bool cpu_llep_prefill_transport_supported;
        bool captured_distributed_overlay_runtime_table;
        bool captured_overlay_route_ledger_uses_grouped_publication;
        MoERuntimeRouteWeightProjection captured_overlay_route_weight_projection;
        std::string &current_batch_llep_plan_node;
        std::string &current_batch_llep_apply_node;
        std::string &current_batch_llep_expert_node;
        std::optional<MoEGPUCurrentBatchLLEPStage::Params> &current_batch_llep_plan_params;
        std::optional<MoEGPUCurrentBatchLLEPStage::Params> &current_batch_llep_apply_params;
        bool &current_batch_llep_payload_sideband_taken;
        std::optional<Qwen4ExpHyperconnectionBuilder::BlockRead> &residual_read;
        const std::shared_ptr<MoERoutedPipelineKernelOwner> &routed_pipeline_kernel_owner;
        TensorBase *routing_indices;
        TensorBase *routing_weights;
        int expert_intermediate;
        TensorBase *moe_output;
        TensorBase *canonical_route_contributions;
        TensorBase *shared_output;
        bool &moe_combined_output_ready;
        std::string &shared_ffn_last;
        std::optional<SharedPrefillCPUInputFork> &shared_prefill_cpu_input_fork;
        std::optional<std::string> &shared_prefill_mapped_dispatch;
        std::optional<std::string> &shared_decode_cpu_packet;
        std::optional<std::string> &shared_decode_ticket_cutpoint;
        bool &routed_overlay_has_distributed_sparse_protocol;
        bool has_shared_expert_branch;
        DeviceId planned_shared_device;
        bool shared_expert_requires_tp_allreduce;
        CanonicalMoEPublicationLowering &canonical_publication_lowering;
        std::optional<DeferredOverlayCombinedPublication> &deferred_overlay_combined_publication;
        std::string &captured_overlay_routed_unit_terminal;
        std::string &captured_overlay_pre_return_terminal;
        std::string &captured_overlay_first_return_node;

        std::function<bool(DeviceId)> forceDecodeEquivalentMoEVerifier;
        std::function<bool(DeviceId)> forceGroupedSharedMoEVerifierPrefill;
        std::function<std::string()> graphRebalanceDomainKey;
        std::function<std::vector<TPAllreduceSidebandWorkspaceBinding>()> takeCurrentBatchLLEPPayloadSideband;
        std::function<void(std::vector<TPAllreduceSidebandWorkspaceBinding> &,
                           std::vector<TPAllreduceSidebandWorkspaceBinding>)> appendRebalanceSidebands;
        std::function<std::vector<TPAllreduceSidebandWorkspaceBinding>()> takeGraphRebalanceSidebandsForAllreduce;
        std::function<void(const std::string &, std::string &)> maybeAddGraphRebalancePayloadStageAfterSideband;
        std::function<bool()> needsMoEParticipantAllreduce;
        std::function<MoEExpertComputeStage::Params(TensorBase *, BufferId, std::vector<bool>, DeviceId)> makeExpertParams;
        std::function<bool(MoEExpertComputeStage::Params &, DeviceId, const std::string &, const std::string &, const MoEExpertOwnerParticipant *)> prepareExpertParams;
        std::function<void(const MoEExpertOwnerMap &, int, const MoEExpertComputeStage::Params &, const std::string &)> registerInitialOverlayParticipantBank;
    };

} // namespace llaminar2
