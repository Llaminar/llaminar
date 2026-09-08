/**
 * @file Qwen35MoEGraph.cpp
 * @brief Qwen 3.5 MoE production graph builder implementation.
 *
 * The builder keeps participant-local compute declarative while lowering
 * explicit sparse dispatch and return collective boundaries for heterogeneous
 * expert-overlay deployments. Dense continuation remains local to each graph;
 * cross-rank expert work is addressed by logical participant identity so CUDA,
 * ROCm, and CPU/NUMA endpoints can share one production model topology.
 */

#include "Qwen35MoEGraph.h"
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
#include "../../execution/compute_stages/stages/MoERankBatchSparseStages.h"
#include "../../execution/compute_stages/stages/MoESparseDispatchStage.h"
#include "../../execution/compute_stages/stages/MoESparseReturnReduceStage.h"
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
#include "../../tensors/NativeVnniFormatInfo.h"
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
         * Version 5 excludes epoch-ticketed durable placement entirely. Expert
         * residency is model-lifetime state owned by the live RCU authority,
         * whereas a prefix block is request state. Version 4 could snapshot a
         * main table and each MTP sidecar independently, then restore their
         * embedded active_bank fields without updating the shared device epoch
         * selector. Rejecting it prevents a cache hit from creating two
         * placement authorities or rewinding model-wide residency.
         */
        constexpr uint32_t kMoEPrefixRuntimeVersion = 5;

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

        bool readU32(const std::vector<uint8_t> &in, size_t &offset, uint32_t &value)
        {
            if (offset + 4 > in.size())
                return false;
            value = 0;
            for (int i = 0; i < 4; ++i)
                value |= static_cast<uint32_t>(in[offset++]) << (8 * i);
            return true;
        }

        bool readI32(const std::vector<uint8_t> &in, size_t &offset, int32_t &value)
        {
            uint32_t raw = 0;
            if (!readU32(in, offset, raw))
                return false;
            value = static_cast<int32_t>(raw);
            return true;
        }

        bool readU64(const std::vector<uint8_t> &in, size_t &offset, uint64_t &value)
        {
            if (offset + 8 > in.size())
                return false;
            value = 0;
            for (int i = 0; i < 8; ++i)
                value |= static_cast<uint64_t>(in[offset++]) << (8 * i);
            return true;
        }

        bool readString(const std::vector<uint8_t> &in, size_t &offset, std::string &value)
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

        std::vector<int> ownerParticipantsFromMap(
            const MoEExpertOwnerMap &owner_map,
            int layer_idx,
            int num_experts)
        {
            std::vector<int> owners(static_cast<size_t>(std::max(0, num_experts)), -1);
            for (int expert = 0; expert < num_experts; ++expert)
            {
                if (const auto *owner = owner_map.ownerFor(layer_idx, expert))
                    owners[static_cast<size_t>(expert)] = owner->owner_participant;
            }
            return owners;
        }

        /**
         * @brief Project global ExpertOverlay ownership into one domain's IDs.
         *
         * A device-local runtime table describes one collective domain, not
         * every participant in a heterogeneous overlay. Experts owned by a
         * different domain are therefore represented by the explicit external
         * sentinel `-1`; experts in @p domain_name use their domain-local
         * participant index. This keeps decode routing independent of global
         * participant numbering and permits the continuation tier to appear at
         * any integer priority in the overlay.
         *
         * @param owner_map Canonical global ExpertOverlay owner map.
         * @param layer_idx Logical routed-expert layer.
         * @param num_experts Number of logical experts in the layer.
         * @param domain_name Domain represented by the device runtime table.
         * @return One domain-local owner per expert, or `-1` when external.
         */
        std::vector<int> domainLocalOwnerParticipantsFromMap(
            const MoEExpertOwnerMap &owner_map,
            int layer_idx,
            int num_experts,
            const std::string &domain_name)
        {
            std::vector<int> owners(
                static_cast<size_t>(std::max(0, num_experts)), -1);
            for (int expert = 0; expert < num_experts; ++expert)
            {
                const auto *owner = owner_map.ownerFor(layer_idx, expert);
                const auto *participant =
                    owner ? owner_map.participantForId(
                                owner->owner_participant)
                          : nullptr;
                if (!participant ||
                    participant->domain_name != domain_name)
                {
                    continue;
                }
                owners[static_cast<size_t>(expert)] =
                    participant->domain_participant_index;
            }
            return owners;
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
            if (std::any_of(
                    overlay_route_participants.begin(),
                    overlay_route_participants.end(),
                    [](int participant) { return participant < -1; }))
            {
                LOG_ERROR("[Qwen35MoEGraph] " << context
                                              << ": overlay route participant vector contains a value below -1");
                return false;
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
                                                  << " cannot export one uniform NativeVNNI/FP16/BF16/FP32 descriptor family");
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
                                                  << " cannot export one uniform NativeVNNI/FP16/BF16/FP32 descriptor family");
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
                return expert.owner_participant >= 0 ||
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
                         {"layer", std::to_string(layer_idx)},
                         {"snapshot_participants", std::to_string(layer.participant_count)},
                         {"expected_participants", std::to_string(*expected_participant_count)}});
                    LOG_DEBUG("[Qwen35MoEGraph] Skipping prefix-cache MoE runtime state for "
                              << table_key << " layer=" << layer_idx
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

        bool supportsDeviceSideGraphRebalanceTransfer(
            const ILocalTPContext &tp_ctx)
        {
            const auto &moe_env = debugEnv().moe_rebalance;
            if (moe_env.device_rebalance_maintenance_graph)
                return tp_ctx.supportsRawAllgatherOnStreamGraphCapture();
            return false;
        }

        bool isHomogeneousGpuLocalTPRebalanceDomain(
            const ILocalTPContext &tp_ctx,
            DeviceId device,
            int tp_device_idx)
        {
            const int degree = tp_ctx.degree();
            if (!device.is_gpu() ||
                degree <= 1 ||
                degree > static_cast<int>(kDeviceMoEMaxParticipants) ||
                tp_device_idx < 0 ||
                tp_device_idx >= degree ||
                !supportsDeviceSideGraphRebalanceTransfer(tp_ctx))
            {
                return false;
            }

            const CollectiveBackendType backend = tp_ctx.backend();
            if ((device.is_cuda() && backend != CollectiveBackendType::NCCL) ||
                (device.is_rocm() && backend != CollectiveBackendType::RCCL))
            {
                return false;
            }

            const auto &participants = tp_ctx.devices();
            if (static_cast<int>(participants.size()) != degree)
                return false;

            const DeviceType expected_type =
                device.is_cuda() ? DeviceType::CUDA : DeviceType::ROCm;
            for (const auto &participant : participants)
            {
                if (!participant.isLocal() ||
                    !participant.isGPU() ||
                    participant.device_type != expected_type)
                {
                    return false;
                }
            }

            return participants[static_cast<size_t>(tp_device_idx)].toLocalDeviceId() == device;
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
         *         cannot export a valid, geometry-compatible NativeVNNI descriptor.
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

                DeviceNativeVNNIMatrixDesc descriptor{};
                NativeVnniSourceIdentity source_identity{};
                if (!engine->exportNativeVNNIMatrixDesc(descriptor) ||
                    !engine->exportNativeVNNISourceIdentity(source_identity) ||
                    !source_identity.present ||
                    native_vnni_formats::forSourceIdentity(
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
                    ExpertWeightFormat::nativeVnni(source_identity);
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


        /**
         * @brief Immutable graph-lowering facts for the continuation tier.
         *
         * `graph_local_participant` uses the global overlay identity because
         * sparse dispatch and residency banks use that namespace. Device
         * runtime tables instead use the participant's domain-local index;
         * callers obtain it from the owner-map participant record.
         */
        struct RoutedContinuationTopology
        {
            int tier_index = -1;
            const RoutedExpertDomain *domain = nullptr;
            int graph_local_participant = -1;
            bool captured_continuation_device = false;
            bool captured_local_tp = false;
        };

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

            const MoEExpertOwnerMap owner_map =
                MoEExpertOwnerMap::build(plan);
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
                result.domain->routed_compute_policy ==
                    RoutedExpertComputePolicy::Apportioned &&
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

    void Qwen35MoEGraph::resetState(void *execution_stream)
    {
        Qwen35Graph::resetState(execution_stream);
        prefix_runtime_device_rehydration_pending_ = false;

        /* Routing evidence is model/residency-epoch state, not request state.
         * The maintenance authority alone rotates its RCU banks. Resetting it
         * here used to erase short-request evidence and could race an async
         * device drain while prefix or KV state was being cleared. */

        for (auto &[key, table] : moe_runtime_tables_)
        {
            (void)key;
            /*
             * Canonical main and MTP tables are model-lifetime residency state.
             * An LLEP child also carries the durable ticket, but its embedded
             * override banks are request-local and must return to their
             * immutable template before the next request follows the parent.
             */
            if (table &&
                (!table->usesOverlayEpochTicket() ||
                 table->overlayPlacementSource() != nullptr))
                table->restoreInitialRuntimeState(execution_stream);
        }

        /*
         * Transfer directories are model-lifetime allocation authorities.
         * Their entries may back a durable ticket-selected bank even after a
         * transient CurrentBatchLLEP table has been reset. The device allocator
         * derives liveness from the runtime family and uses compare-and-replace
         * generations, so stale request-local occupants are safely reusable;
         * clearing the shared directory here would instead invalidate durable
         * descriptors still selected by the next request's epoch ticket.
         */
    }

    void Qwen35MoEGraph::resetPrefixCacheRuntimeStateWithoutSnapshot(
        void *execution_stream)
    {
        /*
         * A cache block without a portable MoE payload owns no request-local
         * placement changes. Reset only transient CurrentBatchLLEP tables.
         * Epoch-ticketed placement remains owned by the live model-lifetime RCU
         * authority and is intentionally independent of prefix-cache state.
         */
        Qwen35Graph::resetState(execution_stream);
        prefix_runtime_device_rehydration_pending_ = false;

        /* Prefix reset does not own model-lifetime routing evidence. The
         * residency maintenance authority rotates that state independently. */

        /*
         * Graph-side bindings, auxiliary streams, and transfer-slot
         * directories own model-lifetime pointer identities captured by stage
         * nodes. Prefix reset must preserve those owners. Non-ticketed runtime
         * placement claims are request state; restoreInitialRuntimeState()
         * removes those transient claims while retaining stable pointer
         * identities. Ticketed canonical placement is not reset, while a
         * ticketed LLEP child is reset because its source pointer identifies
         * the durable parent independently of its embedded override banks.
         *
         * Clearing these maps used to free directories underneath existing
         * graph stages and then assume a later graph build would recreate
         * them.  ComputeGraph nodes outlive replay executables, so that
         * assumption left dangling transfer pointers and made prefix/MTP
         * failures dependent on whichever layer reused the freed slot first.
         */

        for (auto &[key, table] : moe_runtime_tables_)
        {
            (void)key;
            if (table &&
                (!table->usesOverlayEpochTicket() ||
                 table->overlayPlacementSource() != nullptr))
                table->restoreInitialRuntimeState(execution_stream);
        }
    }

    /**
     * @brief Locate the domain-wide decode maintenance binding for a device.
     *
     * The graph builder can create two independent classes of graph-side
     * transfer binding when Dynamic residency maintenance and current-batch
     * LLEP are both selected explicitly: one domain-wide decode-maintenance
     * binding and one or more layer-local prefill-transfer bindings. The async
     * maintenance graph is a decode-time control loop, so it must use the
     * decode binding even when a prefill binding was inserted later into the
     * unordered binding map.
     */
    const Qwen35MoEGraph::GraphSideRebalanceBinding *
    Qwen35MoEGraph::findDeviceMoERebalanceMaintenanceBinding(DeviceId device) const
    {
        const GraphSideRebalanceBinding *selected = nullptr;
        for (const auto &[key, binding] : moe_graph_rebalance_bindings_)
        {
            (void)key;
            if (binding.device_id != device)
            {
                continue;
            }
            if (binding.role != GraphSideRebalanceBindingRole::DecodeMaintenance)
            {
                continue;
            }
            if (binding.workspace_name.rfind("moe_device_rebalance_", 0) != 0)
            {
                throw std::runtime_error(
                    "Qwen35 MoE decode maintenance binding for " +
                    device.to_string() +
                    " points at a non-decode workspace: " +
                    binding.workspace_name);
            }
            if (selected)
            {
                throw std::runtime_error(
                    "Qwen35 MoE graph-side rebalance has multiple decode maintenance bindings for " +
                    device.to_string() +
                    "; refusing unordered maintenance graph selection");
            }
            selected = &binding;
        }

        return selected;
    }

    DeviceMoECurrentBatchLLEPEvidenceSource
    Qwen35MoEGraph::deviceMoECurrentBatchLLEPEvidenceSource(
        DeviceId device) const
    {
        DeviceMoECurrentBatchLLEPEvidenceSource selected{};
        const IMoERuntimeTable *selected_table = nullptr;

        for (const auto &[key, binding] : moe_graph_rebalance_bindings_)
        {
            (void)key;
            if (binding.device_id != device ||
                binding.role !=
                    GraphSideRebalanceBindingRole::CurrentBatchLLEPTransfer)
            {
                continue;
            }
            if (!binding.moe_runtime_table ||
                binding.moe_runtime_table->layerCount() <= 0)
            {
                throw std::logic_error(
                    "Qwen35 MoE current-batch LLEP evidence binding has no "
                    "complete runtime table for " +
                    device.to_string());
            }

            const auto *runtime_layers =
                binding.moe_runtime_table->deviceLayerState(0);
            const int layer_count =
                binding.moe_runtime_table->layerCount();
            if (!runtime_layers)
            {
                throw std::logic_error(
                    "Qwen35 MoE current-batch LLEP evidence binding has a "
                    "null device runtime table for " +
                    device.to_string());
            }
            if (selected_table &&
                (selected_table != binding.moe_runtime_table ||
                 selected.runtime_layers_device != runtime_layers ||
                 selected.layer_count != layer_count ||
                 selected.expert_payload_slot_bytes !=
                     binding.collective_payload_slot_bytes))
            {
                throw std::logic_error(
                    "Qwen35 MoE current-batch LLEP bindings disagree on the "
                    "canonical runtime table or expert payload extent for " +
                    device.to_string());
            }
            if (binding.collective_payload_slot_bytes == 0)
            {
                throw std::logic_error(
                    "Qwen35 MoE current-batch LLEP evidence binding has a "
                    "zero expert payload extent for " +
                    device.to_string());
            }

            selected_table = binding.moe_runtime_table;
            selected = DeviceMoECurrentBatchLLEPEvidenceSource{
                .runtime_layers_device = runtime_layers,
                .layer_count = layer_count,
                .expert_payload_slot_bytes =
                    binding.collective_payload_slot_bytes,
            };
        }

        return selected;
    }

    bool Qwen35MoEGraph::finalizeMoEOverlayDeviceControllerRuntime(
        DeviceId device,
        void *publication_stream)
    {
        if (!device.is_gpu() || !publication_stream)
        {
            LOG_ERROR(
                "Qwen35 MoE initial controller runtime publication requires "
                "one exact GPU and non-null stream");
            return false;
        }

        const auto runtime_plan = runtimePlanForGraph(config_);
        const auto &placement_plan = runtime_plan
                                         ? runtime_plan->sourcePlanPtr()
                                         : config_.moe.routed_expert_plan;
        if (!placement_plan ||
            !placement_plan->usesExpertOverlayAuthority())
        {
            return true;
        }

        const MoERuntimeTableIdentity identity{
            .role = MoERuntimeTableRole::MainDecodeDurablePlacement,
            .mtp_depth = -1,
        };
        const auto table_it = moe_runtime_tables_.find(
            moeRuntimeTableKey(device, identity));
        if (table_it == moe_runtime_tables_.end() || !table_it->second ||
            table_it->second->layerCount() <= 0)
        {
            LOG_ERROR(
                "Qwen35 MoE retained controller runtime finalization cannot "
                "resolve its canonical main table on "
                << device.toString());
            return false;
        }

        const auto owner_map = MoEExpertOwnerMap::build(*placement_plan);
        const int local_world_rank = config_.moe.overlay_mpi_ctx
                                         ? config_.moe.overlay_mpi_ctx->rank()
                                         : -1;
        const MoEExpertOwnerParticipant *selected = nullptr;
        for (const auto &participant : owner_map.participants())
        {
            if (participant.device != device ||
                (participant.world_rank_known && local_world_rank >= 0 &&
                 participant.world_rank != local_world_rank))
            {
                continue;
            }
            if (selected)
            {
                LOG_ERROR(
                    "Qwen35 MoE retained controller runtime maps one device "
                    "to multiple local overlay participants: "
                    << device.toString());
                return false;
            }
            selected = &participant;
        }
        if (!selected || selected->participant_id < 0 ||
            selected->domain_participant_index < 0 ||
            !selected->world_rank_known)
        {
            LOG_ERROR(
                "Qwen35 MoE retained controller runtime cannot resolve its "
                "local overlay participant on "
                << device.toString());
            return false;
        }

        const auto weight_manager =
            model_ctx_ ? model_ctx_->concreteWeightManager() : nullptr;
        if (!weight_manager || config_.moe.num_experts <= 0 ||
            config_.moe.top_k <= 0 || config_.d_model <= 0 ||
            config_.moe.intermediate_size <= 0)
        {
            LOG_ERROR(
                "Qwen35 MoE retained controller runtime has incomplete model "
                "geometry or no prepared-engine registry");
            return false;
        }

        auto *const table = table_it->second.get();
        auto &registry = weight_manager->expertGemmRegistry();
        std::uint64_t finalized_layers = 0u;
        for (int layer = 0; layer < table->layerCount(); ++layer)
        {
            const bool publication_missing =
                table->decodeRuntimePublicationRequired(layer) ||
                table->hostLayerState(layer).active_epoch == 0u;
            const auto expert_mask = owner_map.expertMaskForParticipant(
                layer,
                selected->participant_id,
                config_.moe.num_experts);
            std::vector<ITensorGemm *> gate_gemms;
            std::vector<ITensorGemm *> up_gemms;
            std::vector<ITensorGemm *> down_gemms;
            (void)registry.populateExpertEnginesForParticipant(
                selected->domain_name,
                device,
                selected->world_rank,
                selected->domain_participant_index,
                layer,
                config_.moe.num_experts,
                gate_gemms,
                up_gemms,
                down_gemms);

            bool local_payload_complete =
                gate_gemms.size() ==
                    static_cast<std::size_t>(config_.moe.num_experts) &&
                up_gemms.size() ==
                    static_cast<std::size_t>(config_.moe.num_experts) &&
                down_gemms.size() ==
                    static_cast<std::size_t>(config_.moe.num_experts);
            for (int expert = 0;
                 local_payload_complete && expert < config_.moe.num_experts;
                 ++expert)
            {
                if (!expert_mask[static_cast<std::size_t>(expert)])
                    continue;
                local_payload_complete =
                    gate_gemms[static_cast<std::size_t>(expert)] != nullptr &&
                    up_gemms[static_cast<std::size_t>(expert)] != nullptr &&
                    down_gemms[static_cast<std::size_t>(expert)] != nullptr;
            }
            if (!local_payload_complete)
            {
                LOG_ERROR(
                    "Qwen35 MoE retained controller runtime cannot resolve "
                    "every local prepared expert for layer "
                    << layer << " participant="
                    << selected->participant_id << " device="
                    << device.toString());
                return false;
            }

            const auto domain_local_owners =
                domainLocalOwnerParticipantsFromMap(
                    owner_map,
                    layer,
                    config_.moe.num_experts,
                    selected->domain_name);
            const auto overlay_route_participants =
                ownerParticipantsFromMap(
                    owner_map,
                    layer,
                    config_.moe.num_experts);
            if (!initializeMaskedLocalDecodeRuntimeTable(
                    table,
                    layer,
                    config_.moe.num_experts,
                    config_.moe.top_k,
                    config_.d_model,
                    config_.moe.intermediate_size,
                    expert_mask,
                    selected->domain_participant_index,
                    static_cast<int>(
                        owner_map.participantIdsForTier(
                            selected->tier_idx)
                            .size()),
                    domain_local_owners,
                    overlay_route_participants,
                    gate_gemms,
                    up_gemms,
                    down_gemms,
                    publication_stream,
                    /*allow_existing_dynamic_bank=*/true,
                    "complete retained ExpertOverlay controller runtime"))
            {
                return false;
            }
            if (publication_missing)
                ++finalized_layers;
        }

        try
        {
            /*
             * A host-side recipe can be complete even when the graph that
             * first referenced a dormant NextN layer has never launched.
             * Close setup with one explicit stream-ordered authority handoff
             * before the device controller scans or mutates the family.
             */
            table->sealAndPublishCompleteInitialRuntimeState(
                publication_stream);
        }
        catch (const std::exception &error)
        {
            LOG_ERROR(
                "Qwen35 MoE retained controller runtime could not seal and "
                "publish every physical model layer on "
                << device.toString() << ": " << error.what());
            return false;
        }

        PerfStatsCollector::addCounter(
            "moe_overlay_controller",
            "retained_runtime_layers_finalized",
            static_cast<double>(finalized_layers),
            "model_setup",
            device.toString(),
            {{"participant", std::to_string(selected->participant_id)},
             {"retained_layers", std::to_string(table->layerCount())},
             {"publication_stream", "controller_ordered"},
             {"blocking", "false"}});
        return true;
    }

    void Qwen35MoEGraph::retireBorrowedExecutionStreams(DeviceId device)
    {
        /* One graph builder may retain tables for main decode, prefill, every
         * MTP depth, and transient LLEP roles. They all borrow streams from the
         * same device-context lifetime, so retirement is a device-wide terminal
         * transition across the complete retained family rather than a guess at
         * whichever table most recently published a histogram. */
        for (auto &[key, table] : moe_runtime_tables_)
        {
            (void)key;
            if (table && table->deviceId() == device)
                table->retireRuntimeHistogramProducerStreams();
        }
    }

    MoEOverlayDeviceControllerRuntimeBinding
    Qwen35MoEGraph::deviceMoEOverlayControllerRuntimeBinding(
        DeviceId device) const
    {
        if (!device.is_gpu())
            return {};
        const auto runtime_plan = runtimePlanForGraph(config_);
        const auto &placement_plan = runtime_plan
                                         ? runtime_plan->sourcePlanPtr()
                                         : config_.moe.routed_expert_plan;
        if (!placement_plan ||
            !placement_plan->usesExpertOverlayAuthority())
        {
            return {};
        }

        const MoERuntimeTableIdentity identity{
            .role = MoERuntimeTableRole::MainDecodeDurablePlacement,
            .mtp_depth = -1,
        };
        const auto table_it = moe_runtime_tables_.find(
            moeRuntimeTableKey(device, identity));
        if (table_it == moe_runtime_tables_.end() || !table_it->second ||
            table_it->second->layerCount() <= 0)
        {
            return {};
        }

        const auto owner_map = MoEExpertOwnerMap::build(*placement_plan);
        const int local_world_rank = config_.moe.overlay_mpi_ctx
                                         ? config_.moe.overlay_mpi_ctx->rank()
                                         : -1;
        const MoEExpertOwnerParticipant *selected = nullptr;
        for (const auto &participant : owner_map.participants())
        {
            if (participant.device != device ||
                (participant.world_rank_known && local_world_rank >= 0 &&
                 participant.world_rank != local_world_rank))
            {
                continue;
            }
            if (selected)
            {
                throw std::logic_error(
                    "Qwen35 MoE controller runtime device maps to multiple local overlay participants: " +
                    device.to_string());
            }
            selected = &participant;
        }
        if (!selected || selected->participant_id < 0 ||
            selected->domain_participant_index < 0)
        {
            throw std::logic_error(
                "Qwen35 MoE controller runtime cannot resolve its global overlay participant: " +
                device.to_string());
        }

        auto *const table = table_it->second.get();
        const auto &runtime = table->hostLayerState(0);
        auto *const epoch_arena = table->overlayEpochArena();
        if (!epoch_arena || epoch_arena->deviceId() != device)
        {
            throw std::logic_error(
                "Qwen35 MoE controller runtime lost its canonical device epoch arena for " +
                device.to_string());
        }
        MoEOverlayDeviceControllerRuntimeBinding binding{
            .device = device,
            .runtime_layers_device = table->deviceLayerState(0),
            .runtime_table_host = table,
            .service_telemetry_device =
                table->deviceOverlayServiceTelemetry(),
            .service_samples_device =
                table->deviceOverlayServiceTelemetrySample(0),
            .overlay_participant_id = selected->participant_id,
            .domain_participant_id = runtime.participant_id,
            .domain_participant_count = runtime.participant_count,
            .layer_count = static_cast<std::uint32_t>(table->layerCount()),
            .expert_count = runtime.expert_count,
            .top_k = runtime.top_k,
            .epoch_control = epoch_arena->deviceControlAddress(),
            .maintenance_epoch =
                epoch_arena->deviceMaintenanceEpochAddress(),
            .maintenance_status =
                epoch_arena->deviceMaintenanceStatusAddress(),
        };
        if (!binding.publicationValid() ||
            binding.domain_participant_id !=
                static_cast<std::uint32_t>(
                    selected->domain_participant_index))
        {
            throw std::logic_error(
                "Qwen35 MoE controller runtime identity disagrees with the overlay owner map for " +
                device.to_string());
        }
        return binding;
    }

    ComputeGraph Qwen35MoEGraph::buildDeviceMoERebalanceMaintenanceGraph(
        DeviceId device)
    {
        const GraphSideRebalanceBinding *selected_binding =
            findDeviceMoERebalanceMaintenanceBinding(device);
        if (!selected_binding)
        {
            return {};
        }

        const GraphSideRebalanceBinding &binding = *selected_binding;
        const bool transfer_slot_mode_enabled =
            deviceMoERebalanceModeUsesTransferSlots(binding.transfer_mode);
        if (!binding.collective_tp_ctx ||
            !binding.moe_runtime_table ||
            (transfer_slot_mode_enabled &&
             (!binding.local_transfer_slots ||
              binding.local_transfer_slot_count == 0 ||
              !binding.transfer_state)))
        {
            throw std::runtime_error(
                "Qwen35 MoE device rebalance maintenance graph has incomplete binding for " +
                device.to_string());
        }

        /*
         * Maintenance is one graph-owned transaction. PlanCopyApply snapshots
         * source bytes before transport, uses the fixed-capacity collective
         * payload lane, and applies arrivals before the terminal event is
         * recorded. The device controller can publish a no-work command, so
         * the host never needs to inspect an edge mask or select a follow-up
         * graph.
         */
        MoEDeviceRebalanceStage::Params params;
        params.device_id = binding.device_id;
        params.tp_ctx = binding.collective_tp_ctx;
        params.moe_runtime_table = binding.moe_runtime_table;
        params.tp_device_idx = binding.tp_device_idx;
        params.config = binding.config;
        params.local_transfer_slots = binding.local_transfer_slots;
        params.local_transfer_slot_count = binding.local_transfer_slot_count;
        params.collective_payload_slot_bytes =
            binding.collective_payload_slot_bytes;
        params.collective_payload_slot_capacity =
            binding.collective_payload_slot_capacity;
        params.stage_name = "moe_device_rebalance_maintenance";
        params.workspace_name = binding.workspace_name;
        params.phase = DeviceMoERebalanceStagePhase::PlanCopyApply;
        params.transfer_mode = binding.transfer_mode;
        params.transfer_state = binding.transfer_state;
        params.join_transfer_stream_after_copy = true;

        auto *canonical_runtime =
            dynamic_cast<DeviceMoERuntimeTable *>(binding.moe_runtime_table);
        if (!canonical_runtime)
        {
            throw std::runtime_error(
                "Qwen35 MoE maintenance binding is not a device runtime table for " +
                device.to_string());
        }
        if (canonical_runtime->usesOverlayEpochTicket())
        {
            const auto epoch_binding =
                deviceMoEOverlayEpochExecutionBinding(device);
            if (!epoch_binding ||
                canonical_runtime->overlayEpochArena() !=
                    epoch_binding.arena.get() ||
                canonical_runtime->overlayPlacementSource() != nullptr)
            {
                throw std::logic_error(
                    "Qwen35 MoE durable maintenance lost its canonical ExpertOverlay epoch authority for " +
                    device.to_string());
            }
            params.overlay_epoch_arena = epoch_binding.arena;
        }

        ComputeGraph graph;
        graph.addNode(
            params.stage_name,
            ComputeStageFactory::createMoEDeviceRebalance(params),
            binding.device_id);
        graph.setTerminalNode(params.stage_name);
        return graph;
    }

    void Qwen35MoEGraph::appendPrefixCacheFingerprintMaterial(PrefixFingerprintMaterial &material) const
    {
        material.moe.push_back({"graph.num_experts", std::to_string(config_.moe.num_experts)});
        material.moe.push_back({"graph.top_k", std::to_string(config_.moe.top_k)});
        material.moe.push_back({"graph.owner_order",
                                routedExpertOwnerOrderToString(config_.moe.owner_order)});
        material.moe.push_back({"graph.owner_participant_index",
                                std::to_string(config_.moe.owner_participant_index)});
        material.moe.push_back({"graph.owner_participant_count",
                                std::to_string(config_.moe.owner_participant_count)});

        if (config_.moe.routed_expert_plan)
        {
            appendExpertOverlayPlanFingerprintFields(
                material.moe,
                *config_.moe.routed_expert_plan,
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

    bool Qwen35MoEGraph::capturePrefixCacheRuntimeState(std::vector<uint8_t> &state, void *stream)
    {
        /*
         * DeviceMoELayerRuntime itself is not a prefix-cache payload: it owns
         * process-local packed-weight descriptors and transfer-slot pointers.
         * Prefix blocks instead carry portable logical MoE runtime state:
         * owners, residency masks, local-compute masks, replica roles, epochs,
         * and decode histograms.  Restore resolves those logical entries back
         * to descriptors in the live runner and fails hard if required local
         * payloads are not resident.
         */
        state.clear();
        if (moe_runtime_tables_.empty())
            return true;

        struct CapturedTable
        {
            std::string key;
            uint32_t layers = 0;
            uint32_t experts = 0;
            std::vector<DeviceMoEPortableLayerRuntimeState> runtime_layers;
        };

        std::vector<CapturedTable> captured_tables;
        const auto expected_participant_count =
            expectedPrefixRuntimeParticipantCount(config_);
        for (auto &[key, table] : moe_runtime_tables_)
        {
            if (!table)
                continue;
            /*
             * Prefix blocks restore request state; durable ExpertOverlay
             * placement is live model state and may have advanced many epochs
             * since the block was captured. MTP sidecars share that exact main
             * placement authority, so excluding every ticketed table also
             * prevents stale embedded sidecar banks from entering the payload.
             */
            if (table->usesOverlayEpochTicket())
            {
                PerfStatsCollector::addCounter(
                    "prefix_cache",
                    "moe_durable_runtime_state_excluded_from_prefix",
                    1.0,
                    "prefix_cache",
                    config_.default_device.toString(),
                    {{"table", key}});
                continue;
            }
            std::vector<DeviceMoEPortableLayerRuntimeState> runtime_layers;
            if (!table->capturePortableRuntimeState(runtime_layers, stream))
            {
                LOG_ERROR("[Qwen35MoEGraph] Failed to capture prefix-cache MoE runtime state for " << key);
                return false;
            }
            if (runtime_layers.empty())
                continue;
            if (!portableRuntimeStateMatchesPrefixRestoreDomain(
                    key,
                    runtime_layers,
                    expected_participant_count,
                    config_.default_device))
            {
                continue;
            }
            CapturedTable captured;
            captured.key = key;
            captured.layers = static_cast<uint32_t>(runtime_layers.size());
            captured.experts = static_cast<uint32_t>(config_.moe.num_experts);
            captured.runtime_layers = std::move(runtime_layers);
            captured_tables.push_back(std::move(captured));
        }

        if (captured_tables.empty())
            return true;

        appendBytes(state, kMoEPrefixRuntimeMagic, sizeof(kMoEPrefixRuntimeMagic));
        appendU32(state, kMoEPrefixRuntimeVersion);
        appendU32(state, static_cast<uint32_t>(config_.moe.top_k));
        appendU32(state, static_cast<uint32_t>(captured_tables.size()));
        for (const auto &captured : captured_tables)
        {
            appendString(state, captured.key);
            appendU32(state, captured.layers);
            appendU32(state, captured.experts);
            for (const auto &layer : captured.runtime_layers)
            {
                appendU32(state, layer.active_epoch);
                appendU32(state, layer.expert_count);
                appendU32(state, layer.top_k);
                appendU32(state, layer.participant_id);
                appendU32(state, layer.participant_count);
                appendU32(
                    state,
                    layer.requires_device_payload_rehydration);
                for (const auto &expert : layer.experts)
                {
                    appendI32(state, expert.logical_expert_id);
                    appendI32(state, expert.owner_participant);
                    appendI32(state, expert.local_slot);
                    appendU32(state, expert.flags);
                    appendU32(state, static_cast<uint32_t>(expert.local_compute));
                    appendU32(state, static_cast<uint32_t>(expert.replica_role));
                    appendU32(state, expert.resident_participant_mask);
                }
                for (uint64_t count : layer.selected_histogram)
                    appendU64(state, count);
                for (uint64_t count : layer.local_histogram)
                    appendU64(state, count);
            }
        }

        PerfStatsCollector::addCounter(
            "prefix_cache",
            "moe_portable_runtime_state_captures",
            1.0,
            "prefix_cache",
            config_.default_device.toString(),
            {{"tables", std::to_string(captured_tables.size())},
             {"bytes", std::to_string(state.size())}});
        return true;
    }

    PrefixCacheRuntimeRestoreResult
    Qwen35MoEGraph::restorePrefixCacheRuntimeState(
        const std::vector<uint8_t> &state,
        void *stream)
    {
        prefix_runtime_device_rehydration_pending_ = false;
        if (state.empty())
        {
            return PrefixCacheRuntimeRestoreResult{
                .restored = true,
            };
        }
        if (state.size() < sizeof(kMoEPrefixRuntimeMagic) ||
            std::memcmp(state.data(), kMoEPrefixRuntimeMagic, sizeof(kMoEPrefixRuntimeMagic)) != 0)
        {
            LOG_ERROR("[Qwen35MoEGraph] Refusing obsolete prefix-cache MoE runtime state: "
                      "MoE placement snapshots must not serialize runtime-local device descriptors");
            return {};
        }

        size_t offset = sizeof(kMoEPrefixRuntimeMagic);
        uint32_t version = 0;
        uint32_t top_k = 0;
        uint32_t table_count = 0;
        if (!readU32(state, offset, version) ||
            !readU32(state, offset, top_k) ||
            !readU32(state, offset, table_count))
        {
            LOG_ERROR("[Qwen35MoEGraph] Malformed prefix-cache MoE runtime state header");
            return {};
        }
        if (version != kMoEPrefixRuntimeVersion)
        {
            LOG_ERROR("[Qwen35MoEGraph] Unsupported prefix-cache MoE runtime state version "
                      << version);
            return {};
        }
        if (top_k != static_cast<uint32_t>(config_.moe.top_k))
        {
            LOG_ERROR("[Qwen35MoEGraph] Prefix-cache MoE histogram top-k mismatch: blob="
                      << top_k << " graph=" << config_.moe.top_k);
            return {};
        }

        uint32_t restored_tables = 0;
        bool requires_device_rehydration = false;
        bool placement_changed = false;
        for (uint32_t table_idx = 0; table_idx < table_count; ++table_idx)
        {
            std::string key;
            uint32_t layers = 0;
            uint32_t experts = 0;
            if (!readString(state, offset, key) ||
                !readU32(state, offset, layers) ||
                !readU32(state, offset, experts))
            {
                LOG_ERROR("[Qwen35MoEGraph] Malformed prefix-cache MoE portable runtime table header");
                return {};
            }
            if (experts != static_cast<uint32_t>(config_.moe.num_experts))
            {
                LOG_ERROR("[Qwen35MoEGraph] Prefix-cache MoE portable runtime expert-count mismatch for "
                          << key << ": blob=" << experts
                          << " graph=" << config_.moe.num_experts);
                return {};
            }
            auto it = moe_runtime_tables_.find(key);
            if (it == moe_runtime_tables_.end() || !it->second)
            {
                LOG_ERROR("[Qwen35MoEGraph] Prefix-cache MoE portable runtime restore could not find runtime table "
                          << key);
                return {};
            }
            if (it->second->usesOverlayEpochTicket())
            {
                LOG_ERROR("[Qwen35MoEGraph] Prefix-cache MoE runtime state attempted to replace model-lifetime ExpertOverlay placement for "
                          << key);
                return {};
            }
            if (layers != static_cast<uint32_t>(it->second->layerCount()))
            {
                LOG_ERROR("[Qwen35MoEGraph] Prefix-cache MoE portable runtime layer-count mismatch for "
                          << key << ": blob=" << layers
                          << " table=" << it->second->layerCount());
                return {};
            }

            std::vector<DeviceMoEPortableLayerRuntimeState> runtime_layers;
            runtime_layers.resize(static_cast<size_t>(layers));
            for (uint32_t layer_idx = 0; layer_idx < layers; ++layer_idx)
            {
                auto &layer = runtime_layers[static_cast<size_t>(layer_idx)];
                if (!readU32(state, offset, layer.active_epoch) ||
                    !readU32(state, offset, layer.expert_count) ||
                    !readU32(state, offset, layer.top_k) ||
                    !readU32(state, offset, layer.participant_id) ||
                    !readU32(state, offset, layer.participant_count) ||
                    !readU32(
                        state,
                        offset,
                        layer.requires_device_payload_rehydration))
                {
                    LOG_ERROR("[Qwen35MoEGraph] Malformed prefix-cache MoE portable runtime layer header");
                    return {};
                }
                if (layer.expert_count != experts ||
                    layer.top_k != static_cast<uint32_t>(config_.moe.top_k))
                {
                    LOG_ERROR("[Qwen35MoEGraph] Prefix-cache MoE portable runtime layer metadata mismatch for "
                              << key << " layer=" << layer_idx);
                    return {};
                }
                layer.experts.resize(static_cast<size_t>(experts));
                for (uint32_t expert_idx = 0; expert_idx < experts; ++expert_idx)
                {
                    auto &expert = layer.experts[static_cast<size_t>(expert_idx)];
                    int32_t logical_expert_id = -1;
                    int32_t owner_participant = -1;
                    int32_t local_slot = -1;
                    uint32_t local_compute = 0;
                    uint32_t replica_role = 0;
                    if (!readI32(state, offset, logical_expert_id) ||
                        !readI32(state, offset, owner_participant) ||
                        !readI32(state, offset, local_slot) ||
                        !readU32(state, offset, expert.flags) ||
                        !readU32(state, offset, local_compute) ||
                        !readU32(state, offset, replica_role) ||
                        !readU32(state, offset, expert.resident_participant_mask))
                    {
                        LOG_ERROR("[Qwen35MoEGraph] Malformed prefix-cache MoE portable runtime expert payload");
                        return {};
                    }
                    expert.logical_expert_id = logical_expert_id;
                    expert.owner_participant = owner_participant;
                    expert.local_slot = local_slot;
                    expert.local_compute = local_compute != 0u ? 1u : 0u;
                    expert.replica_role = static_cast<uint8_t>(replica_role);
                }
                layer.selected_histogram.assign(static_cast<size_t>(experts), 0ULL);
                layer.local_histogram.assign(static_cast<size_t>(experts), 0ULL);
                layer.prefill_selected_histogram.assign(
                    static_cast<size_t>(experts), 0ULL);
                layer.prefill_local_histogram.assign(
                    static_cast<size_t>(experts), 0ULL);
                layer.grouped_verifier_selected_histogram.assign(
                    static_cast<size_t>(experts), 0ULL);
                layer.grouped_verifier_local_histogram.assign(
                    static_cast<size_t>(experts), 0ULL);
                for (uint32_t expert_idx = 0; expert_idx < experts; ++expert_idx)
                {
                    if (!readU64(state, offset, layer.selected_histogram[static_cast<size_t>(expert_idx)]))
                    {
                        LOG_ERROR("[Qwen35MoEGraph] Malformed prefix-cache MoE portable selected histogram payload");
                        return {};
                    }
                }
                for (uint32_t expert_idx = 0; expert_idx < experts; ++expert_idx)
                {
                    if (!readU64(state, offset, layer.local_histogram[static_cast<size_t>(expert_idx)]))
                    {
                        LOG_ERROR("[Qwen35MoEGraph] Malformed prefix-cache MoE portable local histogram payload");
                        return {};
                    }
                }
            }

            /*
             * Portable version 5 contains only pointer-free, request-transient
             * logical placement. Model-lifetime ExpertOverlay placement is
             * selected by the shared RCU epoch ticket and is rejected above so
             * an old prefix can never rewind main or MTP residency.
             *
             * Transfer-slot descriptors for the remaining request-local state
             * are reconstructed from immutable owner payloads by the dedicated
             * captured rehydration transaction. Blob parsing must never consult
             * a rolling slot directory whose bytes may have been reused since
             * the prefix was harvested.
             */
            const DeviceMoEPortableRuntimeRestoreResult table_restore =
                it->second->restorePortableRuntimeState(runtime_layers, stream);
            if (!table_restore)
            {
                LOG_ERROR("[Qwen35MoEGraph] Prefix-cache MoE portable runtime restore failed for " << key);
                return {};
            }
            placement_changed =
                placement_changed ||
                table_restore.placement_effect ==
                    DeviceMoEPortablePlacementEffect::Changed;
            requires_device_rehydration =
                requires_device_rehydration ||
                table_restore.requires_device_payload_rehydration;
            ++restored_tables;
        }
        if (offset != state.size())
        {
            LOG_ERROR("[Qwen35MoEGraph] Prefix-cache MoE runtime state has trailing bytes");
            return {};
        }

        prefix_runtime_device_rehydration_pending_ =
            requires_device_rehydration;
        LOG_INFO("[Qwen35MoEGraph] Prefix-runtime restore transaction"
                 << " device=" << config_.default_device.toString()
                 << " tables=" << restored_tables
                 << " placement_effect="
                 << (placement_changed ? "changed" : "unchanged")
                 << " device_payload_rehydration="
                 << (prefix_runtime_device_rehydration_pending_
                         ? "pending"
                         : "not_required"));
        PerfStatsCollector::addCounter(
            "prefix_cache",
            "moe_portable_runtime_state_restores",
            1.0,
            "prefix_cache",
            config_.default_device.toString(),
            {{"tables", std::to_string(restored_tables)},
             {"bytes", std::to_string(state.size())},
             {"placement_effect",
              placement_changed ? "changed" : "unchanged"},
             {"device_payload_rehydration",
              requires_device_rehydration ? "required" : "not_required"}});
        return PrefixCacheRuntimeRestoreResult{
            .restored = true,
            .placement_effect =
                placement_changed
                    ? PrefixCacheRuntimePlacementEffect::Changed
                    : PrefixCacheRuntimePlacementEffect::Unchanged,
            .device_rehydration_required = requires_device_rehydration,
        };
    }

    void Qwen35MoEGraph::completePrefixCacheRuntimeStateDeviceRehydration()
    {
        if (!prefix_runtime_device_rehydration_pending_)
        {
            throw std::runtime_error(
                "Qwen35 MoE prefix-runtime device rehydration completion was published without a pending restore transaction");
        }
        prefix_runtime_device_rehydration_pending_ = false;
        PerfStatsCollector::addCounter(
            "prefix_cache",
            "moe_portable_runtime_device_rehydrations",
            1.0,
            "prefix_cache",
            config_.default_device.toString());
    }

    std::string Qwen35MoEGraph::moeRuntimeTableKey(
        DeviceId device,
        const MoERuntimeTableIdentity &identity)
    {
        switch (identity.role)
        {
        case MoERuntimeTableRole::MainDecodeDurablePlacement:
            if (identity.mtp_depth >= 0)
            {
                throw std::invalid_argument(
                    "Durable main-decode MoE runtime identity cannot carry an MTP depth");
            }
            return device.to_string();
        case MoERuntimeTableRole::CurrentBatchLLEPPrefill:
            if (identity.mtp_depth >= 0)
            {
                throw std::invalid_argument(
                    "Current-batch LLEP prefill runtime identity cannot carry an MTP depth");
            }
            return device.to_string() + "#current_batch_llep_prefill";
        case MoERuntimeTableRole::MTPDepth:
            if (identity.mtp_depth < 0)
            {
                throw std::invalid_argument(
                    "MTP MoE runtime identity requires a non-negative depth");
            }
            return device.to_string() + "#mtp_depth" +
                   std::to_string(identity.mtp_depth);
        }
        throw std::logic_error("Unknown MoE runtime table role");
    }

    std::shared_ptr<MoELocalExpertSerialBufferArena>
    Qwen35MoEGraph::localExpertSerialBufferArenaForParticipant(
        DeviceId device,
        int participant,
        size_t required_row_capacity,
        std::optional<DeviceId> cpu_canonical_route_gpu_consumer)
    {
        if (!device.is_valid() || participant < 0 || required_row_capacity == 0 ||
            config_.d_model <= 0 || config_.moe.top_k <= 0 ||
            (cpu_canonical_route_gpu_consumer &&
             (!device.is_cpu() ||
              !cpu_canonical_route_gpu_consumer->is_gpu())))
        {
            throw std::invalid_argument(
                "Qwen35 MoE serial compact-buffer arena requires a valid device, "
                "participant, row capacity, d_model, and top_k");
        }

        /*
         * The orchestrator declares decode, prefill buckets, grouped verifier,
         * and MTP sidecar graphs as one serial family. Plan the largest row
         * geometry once, then allocate a separately coherent verifier-sized
         * family beside it. This retains setup-only stable addresses without
         * paying maximum-prefill transfer volume for decode.
         */
        const int prefill_rows = overlayPrefillSegmentRowCapacity(
            config_, device);
        const int verifier_rows =
            retainsMTPGraphCapacity(config_.mtp)
                ? std::max(
                      1,
                      resolveMTPRetainedTargetQueryRows(config_.mtp))
                : 1;
        const size_t planned_rows = static_cast<size_t>(
            std::max(prefill_rows, verifier_rows));
        const size_t planned_row_capacity = std::max(
            required_row_capacity,
            planned_rows);

        const std::string key =
            device.to_string() + "#overlay_participant" +
            std::to_string(participant);
        auto existing = moe_serial_local_expert_buffer_arenas_.find(key);
        if (existing != moe_serial_local_expert_buffer_arenas_.end())
        {
            if (!existing->second ||
                !existing->second->supports(
                    device,
                    required_row_capacity,
                    config_.d_model,
                    config_.moe.top_k))
            {
                throw std::logic_error(
                    "Qwen35 MoE graph requested compact local-expert storage beyond "
                    "the immutable serial-family plan for " +
                    key + ": requested_rows=" +
                    std::to_string(required_row_capacity) +
                    " planned_rows=" +
                    std::to_string(
                        existing->second
                            ? existing->second->rowCapacity()
                            : 0));
            }
            if (cpu_canonical_route_gpu_consumer)
            {
                /* This is also the immutable endpoint-identity check. The
                 * accessor never binds or registers storage after creation. */
                (void)existing->second->mappedCPUCanonicalRoutes(
                    *cpu_canonical_route_gpu_consumer);
            }
            return existing->second;
        }

        MoELocalExpertSerialBufferArena::Config arena_config;
        arena_config.device_id = device;
        arena_config.row_capacity = planned_row_capacity;
        arena_config.row_capacity_buckets =
            MoELocalExpertSerialBufferArena::powerOfTwoRowBucketsThrough(
                planned_row_capacity);
        arena_config.d_model = config_.d_model;
        arena_config.routing_top_k = config_.moe.top_k;
        if (device.is_cpu())
        {
            arena_config.cpu_canonical_route_storage =
                MoELocalExpertSerialBufferArena::
                    CPUCanonicalRouteStoragePolicy::RetainSerialMaximum;
            arena_config.cpu_canonical_route_gpu_consumer =
                cpu_canonical_route_gpu_consumer;
            arena_config.cpu_grouped_scratch_storage =
                MoELocalExpertSerialBufferArena::
                    CPUGroupedScratchStoragePolicy::RetainSerialMaximum;
            arena_config.num_experts = config_.moe.num_experts;
            arena_config.expert_intermediate =
                config_.moe.intermediate_size;
        }
        arena_config.logical_participant_id = participant;
        arena_config.debug_name =
            "moe_overlay_serial_compact." + key;
        auto arena = std::make_shared<MoELocalExpertSerialBufferArena>(
            std::move(arena_config));
        moe_serial_local_expert_buffer_arenas_.emplace(key, arena);
        return arena;
    }

    std::shared_ptr<MappedHostTransferArena>
    Qwen35MoEGraph::mappedOverlayTicketArenaForDevice(DeviceId device)
    {
        if (!device.is_gpu())
        {
            throw std::invalid_argument(
                "Qwen35 MoE mapped overlay ticket arena requires an exact GPU device");
        }
        const std::string key = device.to_string();
        const auto existing = moe_mapped_ticket_arenas_.find(key);
        if (existing != moe_mapped_ticket_arenas_.end())
        {
            if (!existing->second ||
                existing->second->devices().size() != 1u ||
                existing->second->devices().front() != device)
            {
                throw std::logic_error(
                    "Qwen35 MoE mapped ticket arena endpoint identity changed for " +
                    key);
            }
            return existing->second;
        }

        const std::array<DeviceId, 1> endpoints{device};
        auto arena = TransferEngine::instance().createMappedHostArena(endpoints);
        if (!arena)
        {
            throw std::runtime_error(
                "Qwen35 MoE could not create its mapped overlay ticket arena for " +
                key);
        }
        moe_mapped_ticket_arenas_.emplace(key, arena);
        return arena;
    }

    std::shared_ptr<MoEOverlayCollectiveWorkspace>
    Qwen35MoEGraph::overlayProtocolWorkspaceForParticipant(
        DeviceId graph_device,
        int participant)
    {
        if (!graph_device.is_valid() || participant < 0 ||
            config_.d_model <= 0 || config_.moe.top_k <= 0)
        {
            throw std::invalid_argument(
                "Qwen35 MoE overlay protocol workspace requires a valid graph device, participant, and geometry");
        }

        const size_t planned_rows = static_cast<size_t>(
            overlaySparseProtocolRowCapacity(config_, graph_device));
        const size_t top_k = static_cast<size_t>(config_.moe.top_k);
        if (planned_rows > std::numeric_limits<size_t>::max() / top_k)
        {
            throw std::overflow_error(
                "Qwen35 MoE overlay protocol entry capacity overflows size_t");
        }
        const size_t planned_entries = planned_rows * top_k;
        const std::string key =
            graph_device.to_string() + "#overlay_protocol_p" +
            std::to_string(participant);
        const auto existing =
            moe_serial_overlay_protocol_workspaces_.find(key);
        if (existing != moe_serial_overlay_protocol_workspaces_.end())
        {
            if (!existing->second)
            {
                throw std::logic_error(
                    "Qwen35 MoE overlay protocol workspace cache contains a null arena");
            }
            existing->second->ensureCapacity(
                planned_rows,
                planned_entries,
                config_.d_model,
                config_.moe.top_k,
                DeviceId::cpu());
            return existing->second;
        }

        auto workspace = std::make_shared<MoEOverlayCollectiveWorkspace>(
            MoEOverlayCollectiveWorkspace::FixedCapacityConfig{
                .max_rows = planned_rows,
                .max_entries = planned_entries,
                .d_model = config_.d_model,
                .top_k = config_.moe.top_k,
                .device = DeviceId::cpu(),
                .reuse_policy = MoEOverlayCollectiveWorkspace::
                    StorageReusePolicy::SerialGraphFamily,
            });
        moe_serial_overlay_protocol_workspaces_.emplace(key, workspace);
        return workspace;
    }

    std::shared_ptr<IMoEOverlayRankBatchTransport>
    Qwen35MoEGraph::overlayRankBatchTransportForGroup(
        DeviceId graph_device,
        int tier_index,
        int domain_ordinal,
        int source_world_rank,
        int target_world_rank,
        std::vector<int> participant_ids,
        const MoEExpertOwnerMap &owner_map,
        int source_participant_id)
    {
        if (!config_.moe.overlay_mpi_ctx || !graph_device.is_valid() ||
            tier_index < 0 || domain_ordinal < 0 ||
            source_world_rank < 0 || target_world_rank < 0 ||
            source_world_rank == target_world_rank ||
            participant_ids.empty() || source_participant_id < 0 ||
            !model_ctx_ || !config_.moe.routed_expert_plan)
        {
            throw std::invalid_argument(
                "Qwen35 MoE rank-batch transport requires a complete remote-rank topology");
        }
        std::sort(participant_ids.begin(), participant_ids.end());
        if (participant_ids.front() < 0 ||
            std::adjacent_find(
                participant_ids.begin(), participant_ids.end()) !=
                participant_ids.end())
        {
            throw std::invalid_argument(
                "Qwen35 MoE rank-batch participant ids must be unique and non-negative");
        }

        const size_t planned_rows = static_cast<size_t>(
            overlaySparseProtocolRowCapacity(config_, graph_device));
        const size_t top_k = static_cast<size_t>(config_.moe.top_k);
        const size_t participant_count = participant_ids.size();
        const size_t row_multiplier = std::min(top_k, participant_count);
        if (planned_rows > std::numeric_limits<size_t>::max() / top_k ||
            planned_rows >
                std::numeric_limits<size_t>::max() / row_multiplier)
        {
            throw std::overflow_error(
                "Qwen35 MoE rank-batch aggregate capacity overflows size_t");
        }
        const size_t max_total_rows = planned_rows * row_multiplier;
        const size_t max_total_entries = planned_rows * top_k;

        const auto &overlay_plan = *config_.moe.routed_expert_plan;
        if (static_cast<size_t>(tier_index) >=
            overlay_plan.routed_tiers.size())
        {
            throw std::out_of_range(
                "Qwen35 MoE activation channel tier is outside the frozen plan");
        }
        const auto *const source_descriptor =
            owner_map.participantForId(source_participant_id);
        if (!source_descriptor || !source_descriptor->world_rank_known ||
            source_descriptor->world_rank != source_world_rank ||
            source_descriptor->tier_idx < 0 ||
            static_cast<size_t>(source_descriptor->tier_idx) >=
                overlay_plan.routed_tiers.size())
        {
            throw std::logic_error(
                "Qwen35 MoE activation channel cannot resolve its continuation endpoint");
        }
        const auto resolve_domain_ordinal =
            [&](const std::string &domain_name)
        {
            for (size_t index = 0;
                 index < overlay_plan.domains.size();
                 ++index)
            {
                if (overlay_plan.domains[index].name == domain_name)
                    return static_cast<int>(index);
            }
            throw std::logic_error(
                "Qwen35 MoE activation channel domain is absent from the frozen plan");
        };
        if (resolve_domain_ordinal(
                overlay_plan.routed_tiers[static_cast<size_t>(tier_index)]
                    .domain) != domain_ordinal)
        {
            throw std::logic_error(
                "Qwen35 MoE activation channel target domain ordinal diverged from its tier");
        }
        for (const int participant_id : participant_ids)
        {
            const auto *const participant =
                owner_map.participantForId(participant_id);
            if (!participant || !participant->world_rank_known ||
                participant->world_rank != target_world_rank ||
                participant->tier_idx != tier_index ||
                resolve_domain_ordinal(participant->domain_name) !=
                    domain_ordinal)
            {
                throw std::logic_error(
                    "Qwen35 MoE activation channel target group diverged from the frozen owner map");
            }
        }

        const bool retains_mtp_capacity =
            retainsMTPGraphCapacity(config_.mtp);
        const int max_decode_rows = retains_mtp_capacity
                                        ? std::max(
                                              1,
                                              resolveMTPRetainedTargetQueryRows(
                                                  config_.mtp))
                                        : 1;
        const int max_mtp_draft_depth =
            resolveMTPRetainedDraftCapacity(config_.mtp);
        const auto graph_family =
            resolveMoEOverlayInferenceGraphFamilyIdentity(
                model_ctx_->concreteLoader(),
                model_ctx_->architecture(),
                model_ctx_->totalBlockCount(),
                retains_mtp_capacity
                    ? MoEOverlayMTPGraphFamilyPolicy::RetainModelSidecars
                    : MoEOverlayMTPGraphFamilyPolicy::MainOnly,
                /*graph_family_generation=*/1,
                static_cast<int>(planned_rows),
                max_decode_rows,
                config_.max_request_count,
                max_mtp_draft_depth);
        const auto transaction_topology =
            makeMoEOverlayInferenceTopologyIdentity(
                owner_map,
                graph_family,
                source_world_rank,
                target_world_rank);
        const auto activation_graph_families =
            makeMoEOverlayActivationGraphFamilyManifests(graph_family);
        const MoEOverlayActivationLaneEndpoint source_endpoint{
            .world_rank = source_world_rank,
            .participant_id = source_participant_id,
            .tier_priority =
                overlay_plan
                    .routed_tiers[static_cast<size_t>(
                        source_descriptor->tier_idx)]
                    .priority,
            .domain_ordinal =
                resolve_domain_ordinal(source_descriptor->domain_name),
        };

        std::ostringstream key_stream;
        key_stream << graph_device.to_string()
                   << "#tier" << tier_index
                   << "#domain" << domain_ordinal
                   << "#rank" << source_world_rank << "to"
                   << target_world_rank
                   << "#rows" << planned_rows << "#p";
        for (const int participant : participant_ids)
            key_stream << participant << ',';
        const std::string key = key_stream.str();
        const auto existing = moe_overlay_rank_batch_transports_.find(key);
        if (existing != moe_overlay_rank_batch_transports_.end())
        {
            if (!existing->second ||
                existing->second->participantIds() != participant_ids)
            {
                throw std::logic_error(
                    "Qwen35 MoE rank-batch transport cache topology changed");
            }
            return existing->second;
        }

        const std::string channel_identity =
            makeMoEOverlayRankBatchChannelIdentity(
                tier_index,
                domain_ordinal,
                source_world_rank,
                target_world_rank,
                participant_ids);
        if (resolveMoEOverlayRankBatchTransportKind(
                *config_.moe.overlay_mpi_ctx,
                source_world_rank,
                target_world_rank) ==
            MoEOverlayRankBatchTransportKind::NodeLocalSharedRows)
        {
            if (!config_.moe.rank_batch_transport_registry)
            {
                throw std::logic_error(
                    "Qwen35 MoE node-local activation graph has no preflight transport registry for " +
                    channel_identity);
            }
            auto transport =
                config_.moe.rank_batch_transport_registry->require(
                    channel_identity,
                    source_world_rank,
                    target_world_rank,
                    participant_ids);
            moe_overlay_rank_batch_transports_.emplace(key, transport);
            return transport;
        }

        auto wire_workspace =
            std::make_shared<MoEOverlayRankBatchWireWorkspace>(
                MoEOverlayRankBatchWireWorkspace::Config{
                    .participant_ids = participant_ids,
                    .max_total_rows = max_total_rows,
                    .max_total_entries = max_total_entries,
                    .d_model = config_.d_model,
                    .top_k = config_.moe.top_k,
                });
        auto transport = createMoEOverlayRankBatchTransport(
            MoEOverlayRankBatchTransportConfig{
                .mpi_ctx = config_.moe.overlay_mpi_ctx,
                .source_world_rank = source_world_rank,
                .target_world_rank = target_world_rank,
                .workspace = std::move(wire_workspace),
                .max_rows_per_participant = planned_rows,
                .max_entries_per_participant = max_total_entries,
                .d_model = config_.d_model,
                .top_k = config_.moe.top_k,
                .tier_index = tier_index,
                .domain_ordinal = domain_ordinal,
                .channel_identity = channel_identity,
                .transaction_slot_count = 4096,
                .transaction_topology = transaction_topology,
                .source_endpoint = source_endpoint,
                .target_tier_priority =
                    overlay_plan.routed_tiers[static_cast<size_t>(tier_index)]
                        .priority,
                .activation_graph_families = activation_graph_families,
                .local_lanes = [&]
                {
                    std::vector<MoEOverlayActivationLocalLaneBinding> lanes;
                    lanes.reserve(participant_ids.size());
                    for (const int participant_id : participant_ids)
                    {
                        lanes.push_back({
                            .participant_id = participant_id,
                            .device = graph_device,
                        });
                    }
                    return lanes;
                }(),
            });
        moe_overlay_rank_batch_transports_.emplace(key, transport);
        return transport;
    }

    IMoERuntimeTable *Qwen35MoEGraph::moeRuntimeTableForDevice(
        DeviceId device,
        const MoERuntimeTableIdentity &identity,
        int prefill_token_capacity,
        int num_layers_override,
        MoERuntimeHistogramProducerRole histogram_producer_role,
        bool bind_overlay_epoch)
    {
#if !defined(HAVE_ROCM) && !defined(HAVE_CUDA)
        (void)device;
        (void)identity;
        (void)prefill_token_capacity;
        (void)histogram_producer_role;
        (void)bind_overlay_epoch;
        return nullptr;
#else
        // Device-routed grouped MoE (decode + grouped prefill) is supported on any
        // GPU backend: MoERuntimeTable mirrors its placement banks through the
        // generic IBackend abstraction (see MoERuntimeTable.cpp mirrorBackend),
        // so CUDA and ROCm are both valid here. Gating on is_rocm() previously
        // left CUDA without a runtime table, which forced every MoE routing/expert
        // decode stage into a non-capturable manual graph segment.
        const int requested_table_layers =
            num_layers_override > 0 ? num_layers_override : config_.n_layers;
        int table_layers = requested_table_layers;
        if (bind_overlay_epoch &&
            identity.role ==
                MoERuntimeTableRole::MainDecodeDurablePlacement)
        {
            const auto runtime_plan = runtimePlanForGraph(config_);
            const auto &source_plan = runtime_plan
                                          ? runtime_plan->sourcePlanPtr()
                                          : config_.moe.routed_expert_plan;
            if (!source_plan || !source_plan->usesExpertOverlayAuthority())
            {
                throw std::logic_error(
                    "Qwen35 MoE canonical overlay runtime table has no "
                    "routed placement authority");
            }

            /*
             * The durable parent owns banks for the complete weight manifest,
             * including any routed NextN/MTP source layers after the main
             * transformer interval. Child tables may cover a smaller role,
             * but they must never outgrow their canonical parent merely
             * because the main decode graph happened to be built first.
             */
            table_layers =
                source_plan->placementLayerCapacity(requested_table_layers);
        }
        if (!device.is_gpu() || config_.moe.num_experts <= 0 || config_.moe.top_k <= 0 || table_layers <= 0)
            return nullptr;

        /*
         * Graph executables freeze direct route-scratch pointers. Allocate the
         * complete serial execution domain before the first graph is built:
         * ordinary prefill needs the largest configured graph bucket, while
         * grouped verification and request-batched MTP need their flattened
         * target-row capacity. Every role is event/stream ordered by the
         * orchestrator, so one largest-participant arena is both correct and
         * substantially smaller than one allocation per layer and MTP depth.
         */
        const int prefill_graph_rows = overlaySparseProtocolRowCapacity(
            config_, device);
        const int verifier_rows =
            retainsMTPGraphCapacity(config_.mtp)
                ? resolveMTPRetainedTargetQueryRows(config_.mtp)
                : 1;
        const int planned_route_rows =
            std::max({
                1,
                prefill_token_capacity,
                prefill_graph_rows,
                verifier_rows,
            });
        const std::string scratch_key = device.to_string();
        auto scratch_it =
            moe_serial_route_scratch_arenas_.find(scratch_key);
        if (scratch_it == moe_serial_route_scratch_arenas_.end())
        {
            DeviceMoESerialRouteScratchArena::Config scratch_config;
            scratch_config.device_id = device;
            scratch_config.num_experts = config_.moe.num_experts;
            scratch_config.top_k = config_.moe.top_k;
            scratch_config.token_capacity = planned_route_rows;
            scratch_it = moe_serial_route_scratch_arenas_
                             .emplace(
                                 scratch_key,
                                 std::make_shared<
                                     DeviceMoESerialRouteScratchArena>(
                                     scratch_config))
                             .first;
        }
        else if (!scratch_it->second ||
                 scratch_it->second->tokenCapacity() < planned_route_rows)
        {
            throw std::logic_error(
                "Qwen35 MoE graph requested route scratch beyond the immutable "
                "per-device plan for " +
                device.to_string() +
                ": requested=" + std::to_string(planned_route_rows) +
                " planned=" +
                std::to_string(
                    scratch_it->second
                        ? scratch_it->second->tokenCapacity()
                        : 0));
        }

        const std::string key = moeRuntimeTableKey(device, identity);
        DeviceMoERuntimeTable *overlay_placement_source = nullptr;
        if (bind_overlay_epoch)
        {
            if (identity.role == MoERuntimeTableRole::MTPDepth ||
                identity.role ==
                    MoERuntimeTableRole::CurrentBatchLLEPPrefill)
            {
                const MoERuntimeTableIdentity main_identity{
                    .role = MoERuntimeTableRole::MainDecodeDurablePlacement,
                    .mtp_depth = -1,
                };
                const std::string main_key =
                    moeRuntimeTableKey(device, main_identity);
                auto main_it = moe_runtime_tables_.find(main_key);
                if (main_it == moe_runtime_tables_.end() || !main_it->second)
                {
                    /*
                     * Prefill buckets may be captured before the first decode
                     * graph. Materialize the canonical table as topology now;
                     * the normal per-layer prepared-weight publication below
                     * initializes both this parent and the child. This is not
                     * an eager fallback or a second placement recipe: it makes
                     * graph construction order irrelevant while retaining one
                     * model-lifetime bank owner.
                     */
                    (void)moeRuntimeTableForDevice(
                        device,
                        main_identity,
                        prefill_token_capacity,
                        table_layers,
                        MoERuntimeHistogramProducerRole::NotProducer,
                        /*bind_overlay_epoch=*/true);
                    main_it = moe_runtime_tables_.find(main_key);
                    if (main_it == moe_runtime_tables_.end() ||
                        !main_it->second)
                    {
                        throw std::logic_error(
                            "Qwen35 MoE failed to materialize canonical "
                            "ExpertOverlay runtime table " +
                            main_key + " for a child graph");
                    }
                }
                overlay_placement_source = main_it->second.get();
            }
        }
        auto it = moe_runtime_tables_.find(key);
        if (it != moe_runtime_tables_.end())
        {
            if (it->second->usesOverlayEpochTicket() != bind_overlay_epoch)
            {
                throw std::logic_error(
                    "Qwen35 MoE runtime table " + key +
                    " was reused with a different ExpertOverlay epoch-binding policy");
            }
            if (bind_overlay_epoch)
            {
                const auto binding =
                    deviceMoEOverlayEpochExecutionBinding(device);
                if (!binding ||
                    it->second->overlayEpochTicket() !=
                        binding.arena->requestTicket(binding.request_slot))
                {
                    throw std::logic_error(
                        "Qwen35 MoE runtime table " + key +
                        " does not share the canonical ExpertOverlay request ticket");
                }
            }
            if (it->second->overlayPlacementSource() !=
                overlay_placement_source)
            {
                throw std::logic_error(
                    "Qwen35 MoE runtime table " + key +
                    " was reused with a different canonical overlay placement authority");
            }
            if (prefill_token_capacity > 0)
                it->second->ensurePrefillRouteScratchCapacity(prefill_token_capacity);
            registerRuntimeTableHistogramSyncIfNeeded(
                key,
                it->second.get(),
                histogram_producer_role);
            if (identity.role == MoERuntimeTableRole::MTPDepth)
            {
                PerfStatsCollector::addCounter(
                    "mtp",
                    "moe_mtp_sidecar_runtime_table_reuses",
                    1.0,
                    "graph",
                    device.to_string(),
                    {{"runtime_table", key},
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
        table_config.grouped_verifier_histogram_publication =
            identity.role ==
                    MoERuntimeTableRole::MainDecodeDurablePlacement &&
                    config_.moe.rebalance_config.mode !=
                        MoERebalanceRuntimeMode::Off
                ? GroupedVerifierHistogramPublicationMode::AcceptedRows
                : GroupedVerifierHistogramPublicationMode::Disabled;
        if (bind_overlay_epoch &&
            config_.moe.rebalance_config.mode ==
                MoERebalanceRuntimeMode::Dynamic)
        {
            if (!config_.moe.expert_overlay_residency_authority)
            {
                throw std::logic_error(
                    "Qwen35 MoE Dynamic runtime table has no ExpertOverlay residency authority");
            }
            const auto &catalog =
                config_.moe.expert_overlay_residency_authority
                    ->economyLayerCatalog();
            if (!catalog)
            {
                throw std::logic_error(
                    "Qwen35 MoE Dynamic runtime table has no canonical economy layer catalog");
            }
            table_config.overlay_service_telemetry_coverage =
                MoEOverlayServiceTelemetryCoverage::
                    CatalogStratifiedSample;
            table_config.overlay_service_telemetry_catalog = catalog;
        }
        table_config.prefill_token_capacity = planned_route_rows;
        table_config.deferred_verifier_token_capacity =
            identity.role == MoERuntimeTableRole::MainDecodeDurablePlacement &&
                    retainsMTPGraphCapacity(config_.mtp)
                ? verifier_rows
                : 0;
        table_config.serial_route_scratch_arena = scratch_it->second;
        if (bind_overlay_epoch)
        {
            const auto binding =
                deviceMoEOverlayEpochExecutionBinding(device);
            if (!binding)
            {
                throw std::logic_error(
                    "Qwen35 MoE overlay runtime table requested an empty epoch binding on " +
                    device.to_string());
            }
            table_config.overlay_epoch_arena = binding.arena;
            table_config.overlay_epoch_ticket_slot = binding.request_slot;
            table_config.overlay_placement_source =
                overlay_placement_source;
        }

        auto table = std::make_unique<MoERuntimeTable>(table_config);
        IMoERuntimeTable *ptr = table.get();
        registerRuntimeTableHistogramSyncIfNeeded(
            key,
            ptr,
            histogram_producer_role);
        moe_runtime_tables_.emplace(key, std::move(table));
        if (identity.role == MoERuntimeTableRole::MTPDepth)
        {
            PerfStatsCollector::addCounter(
                "mtp",
                "moe_mtp_sidecar_runtime_table_creations",
                1.0,
                "graph",
                device.to_string(),
                {{"runtime_table", key},
                 {"layers", std::to_string(table_layers)},
                 {"num_experts", std::to_string(config_.moe.num_experts)},
                 {"top_k", std::to_string(config_.moe.top_k)},
                 {"histogram_sync", histogram_producer_role ==
                                            MoERuntimeHistogramProducerRole::ProductionDecode
                                        ? "enabled"
                                        : "disabled"}});
        }
        return ptr;
#endif
    }

    DeviceMoEOverlayEpochExecutionBinding
    Qwen35MoEGraph::deviceMoEOverlayEpochExecutionBinding(DeviceId device)
    {
#if !defined(HAVE_ROCM) && !defined(HAVE_CUDA)
        (void)device;
        return {};
#else
        const auto runtime_plan = runtimePlanForGraph(config_);
        const auto &source_plan = runtime_plan
                                      ? runtime_plan->sourcePlanPtr()
                                      : config_.moe.routed_expert_plan;
        const bool overlay_requested =
            source_plan && source_plan->usesExpertOverlayAuthority();
        if (!overlay_requested)
            return {};
        if (!device.is_gpu())
        {
            throw std::invalid_argument(
                "Qwen35 MoE ExpertOverlay epoch binding requires a GPU participant, got " +
                device.to_string());
        }

        std::optional<MoEOverlayDeviceControllerParticipantBinding>
            retirement_readiness_controller;
        if (config_.moe.device_controller_fabric)
        {
            for (const int participant_id :
                 config_.moe.device_controller_fabric->localParticipantIds())
            {
                const auto participant =
                    config_.moe.device_controller_fabric
                        ->participantBinding(participant_id);
                if (participant.device != device)
                    continue;
                if (retirement_readiness_controller ||
                    !participant.valid() || !participant.controller ||
                    !participant.lifetime)
                {
                    throw std::logic_error(
                        "Qwen35 MoE device maps ambiguously or incompletely to the topology-wide admission fabric: " +
                        device.to_string());
                }
                retirement_readiness_controller = participant;
            }
            if (!retirement_readiness_controller)
            {
                throw std::logic_error(
                    "Qwen35 MoE device has no local topology-wide admission binding: " +
                    device.to_string());
            }
        }

        const std::string key = device.to_string();
        auto existing = moe_overlay_epoch_arenas_.find(key);
        if (existing != moe_overlay_epoch_arenas_.end())
        {
            if (!existing->second ||
                existing->second->deviceId() != device ||
                existing->second->requestSlotCapacity() != 1u)
            {
                throw std::logic_error(
                    "Qwen35 MoE ExpertOverlay epoch arena changed identity for " + key);
            }
            return {
                .arena = existing->second,
                .request_slot = 0u,
                .retirement_readiness_controller =
                    retirement_readiness_controller,
            };
        }

        /*
         * Every runtime table begins with bank zero empty, prepares epoch one
         * into its inactive bank, and flips that table to bank one during model
         * graph construction.  Publishing the epoch selector with the same
         * initial bank makes a subsequently acquired ticket valid for every
         * main and MTP layer without a setup-only host shadow.
         */
        DeviceMoEOverlayEpochArena::Config arena_config;
        arena_config.device_id = device;
        arena_config.initial_epoch = 1u;
        arena_config.initial_bank = 1u;
        arena_config.request_slot_capacity = 1u;
        if (config_.moe.device_controller_fabric)
        {
            const auto &participant =
                *retirement_readiness_controller;
                arena_config.external_admission_epoch =
                    &participant.controller->admission_epoch;
                arena_config.external_admission_lifetime =
                    participant.lifetime;
                if (participant.inference_epoch_member)
                {
                    if (!participant.inference_epoch_record)
                    {
                        throw std::logic_error(
                            "Qwen35 MoE continuation participant has no transaction epoch barrier: " +
                            device.to_string());
                    }
                    arena_config.admission_barrier = {
                        .record = participant.inference_epoch_record,
                        .participant_id = static_cast<std::uint32_t>(
                            participant.participant_id),
                    };
                }
        }
        auto arena =
            std::make_shared<DeviceMoEOverlayEpochArena>(arena_config);
        moe_overlay_epoch_arenas_.emplace(key, arena);
        return {
            .arena = std::move(arena),
            .request_slot = 0u,
            .retirement_readiness_controller =
                retirement_readiness_controller,
        };
#endif
    }

    void Qwen35MoEGraph::registerRuntimeTableHistogramSyncIfNeeded(
        const std::string &key,
        IMoERuntimeTable *table,
        MoERuntimeHistogramProducerRole histogram_producer_role)
    {
        if (histogram_producer_role !=
                MoERuntimeHistogramProducerRole::ProductionDecode ||
            !table || !config_.moe.decode_histogram)
            return;

        auto *histogram = config_.moe.decode_histogram;
        const std::string sync_key =
            key + "@" + std::to_string(reinterpret_cast<std::uintptr_t>(histogram));
        if (!moe_runtime_histogram_sync_keys_.insert(sync_key).second)
            return;

        const bool overlay_histogram =
            config_.moe.expert_overlay_residency_authority != nullptr;
        if (overlay_histogram)
        {
            /* The heterogeneous ticket boundary already publishes exact
             * ordinary decode/prefill routes into the shared host histogram.
             * GPU state is authoritative only for accepted grouped-verifier
             * rows, whose acceptance remains device-owned. Selecting that one
             * phase prevents duplicate demand while preserving MTP rigor. */
            table->enableAsyncDecodeHistogramDrain(
                RuntimeExpertHistogramSourceMask{false, false, true});
            histogram->registerRuntimeHistogramDrain(
                [table, histogram]()
                {
                    return table->progressAsyncDecodeHistogramDrain(
                        *histogram);
                });
            histogram->registerRuntimeHistogramAdmission(
                [table](RuntimeExpertHistogramAdmission admission)
                {
                    return table->publishAsyncDecodeHistogramAdmission(
                        admission);
                });
        }
        else
        {
            histogram->registerRuntimeHistogramSync(
                [table, histogram]()
                {
                    void *stream = table->decodeHistogramProducerStream();
                    if (!stream)
                        throw std::runtime_error(
                            "[Qwen35MoEGraph] runtime histogram sync requested before a decode producer stream was recorded");
                    return table->syncDecodeHistogramToHost(*histogram, stream);
                });
        }
        PerfStatsCollector::addCounter(
            "moe_rebalance",
            "runtime_histogram_sync_registrations",
            1.0,
            "graph",
            "",
            {{"key", key},
             {"protocol", overlay_histogram
                              ? "async_double_buffered"
                              : "synchronous_legacy"}});
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
        const int shared_intermediate =
            std::max(0, config_.moe.shared_intermediate_size);
        const int max_ffn_intermediate =
            std::max(expert_intermediate, shared_intermediate);
        int top_k = config_.moe.top_k;
        const int mtp_target_query_rows =
            retainsMTPGraphCapacity(config_.mtp)
                ? std::max(
                      1,
                      resolveMTPRetainedTargetQueryRows(config_.mtp))
                : 1;
        const size_t moe_activation_rows = static_cast<size_t>(
            std::max(std::max(1, seq_len), mtp_target_query_rows));

        config.custom_formulas["moe_top_k"] = static_cast<size_t>(top_k);
        const int local_tp_degree =
            config_.tp_ctx && config_.tp_ctx->isLocal()
                ? std::max(1, config_.tp_ctx->degree())
                : 0;
        config.custom_formulas["moe_canonical_publication_slots"] =
            static_cast<size_t>(top_k + std::max(1, local_tp_degree));
        config.custom_formulas["moe_expert_intermediate"] = static_cast<size_t>(expert_intermediate);
        config.custom_formulas["moe_ffn_intermediate_max"] =
            static_cast<size_t>(max_ffn_intermediate);
        /*
         * MTP sidecar graphs reuse the MoE BufferIds so that snapshot and
         * publication naming stay aligned with the main graph, but the sidecar
         * can execute up to `mtp_target_query_rows` verifier rows while normal
         * decode allocates `seq_len == 1`.  Reserve the larger row count for
         * MoE scratch/output tensors so a four-row verifier cannot overrun a
         * one-row main decode buffer.
         */
        config.custom_formulas["moe_activation_rows"] = moe_activation_rows;

        // Add MoE buffer name → BufferId mappings
        config.buffer_name_to_id["moe_expert_indices"] = BufferId::MOE_EXPERT_INDICES;
        config.buffer_name_to_id["moe_expert_weights"] = BufferId::MOE_EXPERT_WEIGHTS;
        config.buffer_name_to_id["moe_combined_output"] = BufferId::MOE_COMBINED_OUTPUT;
        config.buffer_name_to_id["moe_canonical_route_contributions"] =
            BufferId::MOE_CANONICAL_ROUTE_CONTRIBUTIONS;
        config.buffer_name_to_id["moe_shared_expert_output"] = BufferId::MOE_SHARED_EXPERT_OUTPUT;
        config.buffer_name_to_id["moe_gate_scratch"] = BufferId::MOE_GATE_SCRATCH;
        config.buffer_name_to_id["moe_up_scratch"] = BufferId::MOE_UP_SCRATCH;

        LOG_DEBUG("[Qwen35MoEGraph::getResolverConfig] MoE formulas: "
                  << "moe_top_k=" << top_k
                  << ", moe_canonical_publication_slots="
                  << (top_k + std::max(1, local_tp_degree))
                  << ", moe_expert_intermediate=" << expert_intermediate
                  << ", moe_ffn_intermediate_max=" << max_ffn_intermediate
                  << ", moe_activation_rows=" << moe_activation_rows);

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
        DeviceId device,
        void *device_state_publication_stream,
        const int32_t *sequence_lengths_device,
        const int32_t *absolute_position_ids_device)
    {
        if (device.is_gpu() && !device_state_publication_stream)
        {
            throw std::invalid_argument(
                "[Qwen35MoEGraph::buildFFNGraph] GPU graph construction "
                "requires an explicit non-null device-state publication stream");
        }

        const bool has_overlay_authority =
            config_.moe.expert_overlay_residency_authority != nullptr;
        if (has_overlay_authority !=
            config_.moe.usesExpertOverlayDurableResidencyAuthority())
        {
            throw std::logic_error(
                "Qwen35 MoE graph has an inconsistent ExpertOverlayRCU "
                "durable residency authority binding");
        }
        ComputeGraph graph;
        const bool mtp_sidecar_context = mtpGraphContextActive();
        const int mtp_depth_idx = mtpGraphDepthIndex();
        const std::string prefix = ffnGraphStagePrefix(layer_idx);
        std::string ffn_terminal;
        int total_tokens = batch_size * seq_len;
        /*
         * Every lowering of this routed layer—direct continuation compute,
         * retained remote GPU replay, and serial CPU service—shares one
         * semantic graph identity. Compute that identity once so the direct
         * fast path cannot silently classify a one-row MTP predictor as
         * ordinary decode while endpoint paths publish grouped-verifier
         * evidence.
         */
        const MoEOverlayServicePhaseHint routed_service_phase =
            mtp_sidecar_context || config_.grouped_mtp_verifier ||
                    config_.live_mtp_request_batch_condition
                ? MoEOverlayServicePhaseHint::GroupedVerifier
                : (total_tokens == 1
                       ? MoEOverlayServicePhaseHint::Decode
                       : MoEOverlayServicePhaseHint::Prefill);
        if (layer_idx == 0 && mirroredLayerDiagnosticsEnabled())
        {
            LOG_INFO(
                "[MirroredLayerDiagnostics] graph_build device="
                << device.toString()
                << " total_tokens=" << total_tokens
                << " mtp_sidecar=" << (mtp_sidecar_context ? "true" : "false")
                << " grouped_mtp_verifier="
                << (config_.grouped_mtp_verifier ? "true" : "false")
                << " all_position_logits="
                << (config_.compute_all_position_logits ? "true" : "false")
                << " live_request_batch_condition="
                << (config_.live_mtp_request_batch_condition ? "true" : "false")
                << " checkpoint_graph="
                << (!mtp_sidecar_context &&
                            !config_.grouped_mtp_verifier
                        ? "true"
                        : "false"));
        }
        auto overlay_runtime_plan = runtimePlanForGraph(config_);
        const auto &overlay_plan = overlay_runtime_plan
                                       ? overlay_runtime_plan->sourcePlanPtr()
                                       : config_.moe.routed_expert_plan;
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

        const bool overlay_requested =
            overlay_plan && overlay_plan->usesExpertOverlayAuthority();
        const RoutedExpertLayerPlacement *overlay_placement = overlay_requested
                                                            ? findExpertOverlayPlacement(*overlay_plan, layer_idx)
                                                            : nullptr;
        const bool use_expert_overlay = overlay_requested && overlay_placement &&
                                        isUsableExpertOverlayPlacement(*overlay_placement,
                                                                       *overlay_plan,
                                                                       config_.moe.num_experts,
                                                                       layer_idx);

        if (overlay_requested && !use_expert_overlay)
        {
            throw std::runtime_error(
                "Qwen35 MoE expert overlay was requested but no usable placement exists for layer " +
                std::to_string(layer_idx) +
                "; refusing to lower the request through the non-overlay routed path");
        }

        /*
         * Tiered ExpertOverlay routes every expert parent through the prepared
         * participant registry. A continuation rank may own no local expert
         * slice at all, so a raw 3-D parent is intentionally absent from its
         * model binding view. The router remains a normal immutable tensor;
         * only the routed GEMM parents are registry-owned.
         */
        if (!layer.moe_gate || (!layer.moe_gate_exps && !use_expert_overlay))
        {
            return Qwen35Graph::buildFFNGraph(
                layer,
                buffers,
                layer_idx,
                seq_len,
                batch_size,
                device,
                device_state_publication_stream,
                sequence_lengths_device,
                absolute_position_ids_device);
        }

        /*
         * Verifier batches are tiny (draft depth + bonus row), but the main-model
         * verifier is stateful: its row logits must match serial decode and the
         * accepted row may later publish KV/GDN/conv state.  The verifier path
         * treats M=1 as a real verifier bucket, but M=1 has no batching economy:
         * it uses the explicit decode-equivalent oracle while multi-row GPU verifier batches use
         * the grouped-prefill machinery.  The router still emits serial-decode
         * top-k rows, and the shared expert uses GEMV verifier-row hooks rather
         * than the older MoE grouped prefill helper.
         */
        auto forceGpuSmallMMoESidecarPrefill = [&](DeviceId candidate)
        {
            return (candidate.is_cuda() || candidate.is_rocm()) &&
                   total_tokens > 1 &&
                   mtp_sidecar_context;
        };
        auto forceGpuSmallMMainVerifierPrefill = [&](DeviceId candidate)
        {
            return (candidate.is_cuda() || candidate.is_rocm()) &&
                   total_tokens > 1 &&
                   config_.grouped_mtp_verifier &&
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
             * serial decode rows.  GPU M=1 has no grouped economy to recover,
             * so it stays on ordinary production decode routing. Multi-row GPU verifier batches
             * still uses the decode-equivalent grouped router so the economical
             * expert kernels receive exactly the serial top-k rows.
            */
            const bool gpu_grouped_verifier_rows =
                (candidate.is_cuda() || candidate.is_rocm()) && total_tokens > 1;
            return (candidate.is_cpu() || gpu_grouped_verifier_rows) &&
                   total_tokens >= 1 &&
                   config_.grouped_mtp_verifier &&
                   config_.compute_all_position_logits &&
                   !mtp_sidecar_context;
        };
        auto forceDecodeEquivalentMoEVerifier = [&](DeviceId candidate)
        {
            return candidate.is_cpu() &&
                   total_tokens >= 1 &&
                   config_.grouped_mtp_verifier &&
                   config_.compute_all_position_logits &&
                   !mtp_sidecar_context;
        };
        LayerWeightBindings layer_bindings = layerWeightBindingsForGraph(layer_idx);

        IMoERuntimeTable *moe_runtime_table = nullptr;
        const auto &rocm_env = debugEnv().rocm;
        /*
         * Grouped MoE verifier rows are now a required production path rather
         * than an advertised optional capability.  Keep backend/shape preflight
         * in the stages, but do not let the old grouped-prefill debug kill switch
         * route publishable verifier rows through a non-grouped substitute.
         */
        auto forceGroupedSharedMoEVerifierPrefill = [&](DeviceId candidate)
        {
            return forceGroupedMoEVerifierPrefill(candidate);
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
        /*
         * Static placement has no histogram consumer. Keep its captured
         * router numerically identical while omitting the global counter
         * atomics and producer-stream publication that only Observe/Dynamic
         * maintenance can use. Device-resident Dynamic remains enabled here:
         * it collects evidence on device even though it deliberately does not
         * register a host drain callback below.
         */
        const bool collect_runtime_histogram =
            !use_mtp_runtime_table &&
            config_.moe.rebalance_config.mode !=
                MoERebalanceRuntimeMode::Off;
        const bool register_runtime_histogram = collect_runtime_histogram;
        const bool local_decode_layer =
            !mtp_sidecar_context &&
            total_tokens == 1 &&
            layer_idx >= config_.pp_layer_offset &&
            layer_idx < config_.pp_layer_offset + config_.n_layers &&
            !config_.compute_all_position_logits;
        /*
         * MTP may execute the main model exclusively through a grouped
         * all-position verifier.  That graph is not prefill: each row is a
         * serial-decode-equivalent candidate and its route kernels own the
         * production decode histograms.  Treat it as a decode maintenance
         * producer so the device-side controller never waits for an M=1 graph
         * that this request is not required to build.
         */
        const bool grouped_main_verifier_layer =
            !mtp_sidecar_context &&
            total_tokens > 1 &&
            config_.grouped_mtp_verifier &&
            config_.compute_all_position_logits &&
            layer_idx >= config_.pp_layer_offset &&
            layer_idx < config_.pp_layer_offset + config_.n_layers;
        const RoutedExpertTier *resolved_local_tp_replicated_tier = nullptr;
        const bool resolved_local_tp_replicated_fast_candidate =
            use_expert_overlay &&
            canUseLocalTPReplicatedFastPath(
                *overlay_plan,
                device,
                &resolved_local_tp_replicated_tier);
        const bool ordinary_prefill_graph =
            total_tokens > 1 &&
            !mtp_sidecar_context &&
            !config_.compute_all_position_logits;
        const bool phase_split_local_tp_apportioned_gpu_prefill =
            resolved_local_tp_replicated_fast_candidate &&
            resolved_local_tp_replicated_tier &&
            ordinary_prefill_graph &&
            isPrefillApportionedDecodeReplicatedTier(
                *overlay_plan,
                *resolved_local_tp_replicated_tier);
        const bool resolved_fully_replicated_local_rows =
            (resolved_local_tp_replicated_fast_candidate &&
             !phase_split_local_tp_apportioned_gpu_prefill) ||
            (!use_expert_overlay &&
             config_.moe.routed_compute_policy ==
                 RoutedExpertComputePolicy::Replicated);
        const RoutedExpertRowExecutionPolicy routed_row_execution_policy =
            resolved_fully_replicated_local_rows
                ? RoutedExpertRowExecutionPolicy::FullyReplicatedLocal
                : RoutedExpertRowExecutionPolicy::ParticipantAssigned;
        const bool device_rebalance_decode_layer =
            local_decode_layer || grouped_main_verifier_layer;
        const bool first_device_rebalance_decode_layer =
            device_rebalance_decode_layer &&
            layer_idx == config_.pp_layer_offset;
        const bool last_device_rebalance_decode_layer =
            device_rebalance_decode_layer &&
            layer_idx == config_.pp_layer_offset + config_.n_layers - 1;
        auto *local_tp_ctx = dynamic_cast<ILocalTPContext *>(config_.tp_ctx);
        auto *global_tp_ctx = dynamic_cast<IGlobalTPContext *>(config_.tp_ctx);
        const bool distributed_expert_overlay =
            use_expert_overlay &&
            config_.moe.overlay_mpi_ctx &&
            config_.moe.overlay_mpi_ctx->world_size() > 1;
        const RoutedContinuationTopology routed_continuation_topology =
            use_expert_overlay
                ? resolveRoutedContinuationTopology(
                      *overlay_plan,
                      device,
                      local_tp_ctx,
                      config_.tp_device_idx,
                      config_.moe.overlay_mpi_ctx != nullptr,
                      config_.moe.overlay_mpi_ctx
                          ? config_.moe.overlay_mpi_ctx->rank()
                          : -1)
                : RoutedContinuationTopology{};
        const bool captured_overlay_continuation =
            routed_continuation_topology
                .captured_continuation_device;
        const bool captured_local_tp_continuation =
            routed_continuation_topology
                .captured_local_tp;
        int hot_replica_cap = config_.moe.hot_expert_cache.resolveCap(
            config_.moe.num_experts,
            /*dynamic_rebalance_enabled=*/true);
        const auto &env = debugEnv();
        const bool prefix_runtime_device_rehydration =
            prefixRuntimeRehydrationGraphActive() &&
            !mtp_sidecar_context &&
            device.is_gpu();
        if (prefix_runtime_device_rehydration &&
            layer_idx == config_.pp_layer_offset)
        {
            LOG_INFO("[Qwen35MoEGraph] Building dedicated prefix-runtime "
                     "device rehydration graph"
                     << " device=" << device.to_string()
                     << " total_tokens=" << total_tokens
                     << " first_layer=" << layer_idx);
        }
        const bool llep_prefill_requested =
            ordinary_prefill_graph &&
            routed_row_execution_policy ==
                RoutedExpertRowExecutionPolicy::ParticipantAssigned &&
            config_.moe.routed_prefill_assignment_policy ==
                RoutedExpertAssignmentPolicy::LeastLoadedResident;
        const uint64_t llep_prefill_routed_rows =
            llep_prefill_requested
                ? static_cast<uint64_t>(std::max(0, total_tokens)) *
                      static_cast<uint64_t>(std::max(0, config_.moe.top_k))
                : 0ULL;
        const uint64_t llep_prefill_min_routed_rows =
            config_.moe.routed_prefill_config
                .least_loaded_min_routed_rows;
        const bool llep_prefill_cost_gate_passed =
            !llep_prefill_requested ||
            llep_prefill_min_routed_rows == 0ULL ||
            llep_prefill_routed_rows >= llep_prefill_min_routed_rows;
        const RoutedExpertDomain *cpu_llep_overlay_domain =
            use_expert_overlay && overlay_plan &&
                    overlay_plan->routed_tiers.size() == 1u
                ? expertDomainForTier(
                      *overlay_plan,
                      overlay_plan->routed_tiers.front())
                : nullptr;
        const bool cpu_llep_overlay_topology_supported =
            cpu_llep_overlay_domain != nullptr &&
            overlay_plan->routed_tiers.front().domain ==
                overlay_plan->continuation_domain &&
            static_cast<int>(cpu_llep_overlay_domain->participants.size()) ==
                (global_tp_ctx ? global_tp_ctx->degree() : 0) &&
            std::all_of(
                cpu_llep_overlay_domain->participants.begin(),
                cpu_llep_overlay_domain->participants.end(),
                [](const GlobalDeviceAddress &participant)
                { return participant.isCPU(); });
        const bool cpu_llep_prefill_transport_supported =
            llep_prefill_requested &&
            llep_prefill_cost_gate_passed &&
            device.is_cpu() &&
            use_expert_overlay &&
            cpu_llep_overlay_topology_supported &&
            config_.moe.expert_overlay_residency_authority != nullptr &&
            config_.moe.expert_overlay_participant_residency != nullptr &&
            config_.moe.overlay_mpi_ctx != nullptr &&
            config_.moe.overlay_mpi_ctx->world_size() > 1 &&
            global_tp_ctx != nullptr &&
            global_tp_ctx->degree() > 1 &&
            global_tp_ctx->myIndex() >= 0 &&
            global_tp_ctx->myIndex() < global_tp_ctx->degree() &&
            cpu_current_batch_llep_executor_ != nullptr;
        const bool gpu_llep_prefill_candidate =
            llep_prefill_requested &&
            llep_prefill_cost_gate_passed &&
            local_tp_ctx != nullptr &&
            device.is_gpu();
        const bool llep_prefill_enabled =
            cpu_llep_prefill_transport_supported ||
            gpu_llep_prefill_candidate;
        const bool llep_prefill_transport_supported =
            gpu_llep_prefill_candidate &&
            local_tp_ctx != nullptr &&
            isHomogeneousGpuLocalTPRebalanceDomain(
                *local_tp_ctx,
                device,
                config_.tp_device_idx);
        const bool prefix_runtime_rehydration_transport_supported =
            prefix_runtime_device_rehydration &&
            local_tp_ctx &&
            isHomogeneousGpuLocalTPRebalanceDomain(
                *local_tp_ctx,
                device,
                config_.tp_device_idx);
        if (gpu_llep_prefill_candidate &&
            !llep_prefill_transport_supported)
        {
            throw std::runtime_error(
                "Qwen35 MoE least-loaded prefill requires "
                "a homogeneous graph-capturable NCCL/RCCL LocalTP domain for " +
                device.to_string());
        }
        if (llep_prefill_requested &&
            llep_prefill_cost_gate_passed &&
            !llep_prefill_enabled)
        {
            throw std::runtime_error(
                "Qwen35 MoE least-loaded prefill has no production "
                "transaction authority for " + device.to_string());
        }
        if (prefix_runtime_device_rehydration &&
            !prefix_runtime_rehydration_transport_supported)
        {
            throw std::runtime_error(
                "Qwen35 MoE prefix-runtime payload rehydration requires a homogeneous graph-capturable NCCL/RCCL LocalTP domain for " +
                device.to_string());
        }
        const RoutedExpertAssignmentPolicy prefill_routed_expert_assignment_policy =
            (routed_row_execution_policy ==
                 RoutedExpertRowExecutionPolicy::ParticipantAssigned &&
             total_tokens > 1 &&
             !llep_prefill_enabled)
                ? RoutedExpertAssignmentPolicy::StaticOwner
                : (grouped_main_verifier_layer || total_tokens == 1
                       ? config_.moe.routed_decode_assignment_policy
                       : config_.moe.routed_prefill_assignment_policy);
        if (env.presence.has("LLAMINAR_MOE_REBALANCE_REPLICAS"))
            hot_replica_cap = std::max(0, env.moe_rebalance.max_replicas);
        const bool dynamic_overlay_residency =
            use_expert_overlay &&
            config_.moe.routed_expert_plan &&
            config_.moe.rebalance_config.mode ==
                MoERebalanceRuntimeMode::Dynamic;
        const bool device_resident_authority =
            dynamic_overlay_residency &&
            config_.moe.usesExpertOverlayDurableResidencyAuthority() &&
            config_.moe.authority_execution ==
                MoEOverlayAuthorityExecutionKind::
                    DeviceResident;
        std::size_t authority_participant_count = 0u;
        if (device_resident_authority && overlay_plan &&
            overlay_plan->routed_tiers.size() == 1u)
        {
            const std::string &authority_domain =
                overlay_plan->routed_tiers.front().domain;
            const auto domain = std::find_if(
                overlay_plan->domains.begin(),
                overlay_plan->domains.end(),
                [&](const RoutedExpertDomain &candidate)
                {
                    return candidate.name == authority_domain;
                });
            if (domain != overlay_plan->domains.end())
                authority_participant_count = domain->participants.size();
        }
        /*
         * A mapped topology-wide controller is the sole authority across all
         * homogeneous groups.  The older domain-local NCCL/RCCL controller is
         * retained only for the single-group special case; composing both
         * would create two policy writers for the same runtime table.  Mapped
         * participants still register their device histograms below so the
         * topology-wide authority consumes the real production route counts.
         */
        const bool device_side_graph_rebalance_candidate =
            device_resident_authority &&
            !config_.moe.device_controller_fabric &&
            device.is_gpu() &&
            local_tp_ctx &&
            isHomogeneousGpuLocalTPRebalanceDomain(
                *local_tp_ctx,
                device,
                config_.tp_device_idx);
        if (device_resident_authority &&
            !config_.moe.device_controller_fabric &&
            device.is_gpu() &&
            device_rebalance_decode_layer &&
            authority_participant_count > 1u &&
            !device_side_graph_rebalance_candidate)
        {
            throw std::runtime_error(
                "Qwen35 MoE single-domain native device-resident ExpertOverlay authority requires one graph-capturable NCCL/RCCL LocalTP maintenance family on " +
                device.to_string());
        }
        /*
         * Current-batch migration is an amortized long-prefill policy. Grouped
         * MTP verifier rows instead split work only across the active bank's
         * resident participants; importing a multi-megabyte expert payload for
         * a handful of speculative rows is categorically uneconomical.
         *
         * Prefix restore remains an independent transport transaction. A
         * verifier graph may therefore own compact transport resources for
         * prefix rehydration while its current-batch assignment mode remains
         * resident-only.
         */
        const bool current_batch_llep_transfer_candidate =
            llep_prefill_transport_supported;
        const bool prefill_llep_transfer_candidate =
            current_batch_llep_transfer_candidate ||
            prefix_runtime_rehydration_transport_supported;
        const GraphSideRebalanceBindingRole prefill_transfer_binding_role =
            current_batch_llep_transfer_candidate
                ? GraphSideRebalanceBindingRole::CurrentBatchLLEPTransfer
                : GraphSideRebalanceBindingRole::PrefixRuntimeRehydrationTransfer;
        const bool graph_rebalance_transport_candidate =
            device_side_graph_rebalance_candidate ||
            prefill_llep_transfer_candidate;
        const bool register_runtime_histogram_for_decode =
            register_runtime_histogram &&
            !device_side_graph_rebalance_candidate;
        const MoERuntimeHistogramWorkload runtime_histogram_workload =
            local_decode_layer
                ? MoERuntimeHistogramWorkload::SerialDecode
                : grouped_main_verifier_layer
                      ? MoERuntimeHistogramWorkload::GroupedMainVerifier
                      : MoERuntimeHistogramWorkload::NonDecode;
        const MoERuntimeHistogramProducerRole runtime_histogram_producer_role =
            selectMoERuntimeHistogramProducerRole(
                runtime_histogram_workload,
                register_runtime_histogram_for_decode);
        const auto has_static_full_local_expert_ownership = [&]()
        {
            /*
             * Device-routed one-token MoE decode captures a full layer-local
             * runtime table: router output, local-compute mask, and prepared
             * gate/up/down descriptors for every logical expert.  LocalTP
             * Expert-id-apportioned runners own only a contiguous expert range, so
             * handing them this table would make the expert stage fail at graph
             * build or, worse, capture a single-device contract for a sharded
             * topology. Tiered overlays and graph-side rebalance use masked
             * runtime tables instead; the full table is only for full-owner
             * lanes.
             */
            if (use_expert_overlay)
                return false;

            if (config_.moe.routed_compute_policy == RoutedExpertComputePolicy::Apportioned)
            {
                if (config_.moe.owner_participant_count > 1)
                {
                    return false;
                }
            }

            if (device.is_gpu() && debugEnv().moe_rebalance.gpu_cache_experts_per_layer > 0)
                return false;

            return true;
        };
        const bool static_full_local_expert_ownership =
            has_static_full_local_expert_ownership();
        const bool masked_local_tp_overlay_decode_runtime_table =
            device.is_gpu() &&
            (total_tokens == 1 || forceGroupedMoEVerifierPrefill(device)) &&
            use_expert_overlay &&
            overlay_plan &&
            canUseLocalTPExpertIdApportionedFastPath(*overlay_plan, device);
        /**
         * @brief Invocation that owns a captured heterogeneous route ledger.
         *
         * The mapped sparse reducer consumes one final domain-local
         * participant for every original router slot.  That publication is a
         * transport invariant, not a Dynamic-only optimization: Static must
         * publish its fixed owner just as Dynamic publishes the owner selected
         * from the request-pinned placement epoch.  Giving each captured shape
         * an explicit role prevents an ordinary prefill graph from binding the
         * ledger pointer while accidentally skipping its producer.
         */
        enum class CapturedOverlayRouteLedgerWorkload : std::uint8_t
        {
            Unbound,
            SerialDecode,
            GroupedVerifier,
            OrdinaryPrefill,
        };

        CapturedOverlayRouteLedgerWorkload
            captured_overlay_route_ledger_workload =
                CapturedOverlayRouteLedgerWorkload::Unbound;
        if (device.is_gpu() && captured_overlay_continuation)
        {
            if (total_tokens == 1)
            {
                captured_overlay_route_ledger_workload =
                    CapturedOverlayRouteLedgerWorkload::SerialDecode;
            }
            else if (forceGroupedMoEVerifierPrefill(device))
            {
                captured_overlay_route_ledger_workload =
                    CapturedOverlayRouteLedgerWorkload::GroupedVerifier;
            }
            else
            {
                captured_overlay_route_ledger_workload =
                    CapturedOverlayRouteLedgerWorkload::OrdinaryPrefill;
            }
        }
        const bool captured_distributed_overlay_runtime_table =
            captured_overlay_route_ledger_workload !=
            CapturedOverlayRouteLedgerWorkload::Unbound;
        const bool captured_overlay_route_ledger_uses_grouped_publication =
            captured_overlay_route_ledger_workload ==
                CapturedOverlayRouteLedgerWorkload::GroupedVerifier ||
            captured_overlay_route_ledger_workload ==
                CapturedOverlayRouteLedgerWorkload::OrdinaryPrefill;
        const MoERuntimeRouteWeightProjection
            captured_overlay_route_weight_projection =
                captured_overlay_route_ledger_workload ==
                        CapturedOverlayRouteLedgerWorkload::SerialDecode
                    ? MoERuntimeRouteWeightProjection::DecodeTopK
                : captured_overlay_route_ledger_uses_grouped_publication
                    ? MoERuntimeRouteWeightProjection::GroupedRouteSlots
                    : MoERuntimeRouteWeightProjection::Unspecified;
        /*
         * A retained heterogeneous Dynamic graph can receive a completely new
         * expert descriptor through the durable ExpertOverlay controller even
         * though it does not use the homogeneous graph-local rebalance
         * transport.  Its capture-time descriptor table is therefore only a
         * workspace seed, never an inference authority.  Decode and grouped
         * verification must materialize descriptors from the request-pinned
         * runtime bank after every admitted epoch, or a correctly promoted
         * expert remains invisible to the continuation-local GEMM.
         */
        const bool dynamic_distributed_overlay_uses_mutable_descriptors =
            dynamic_overlay_residency &&
            captured_distributed_overlay_runtime_table;
        const bool full_local_tp_replicated_overlay_decode_runtime_table =
            device.is_gpu() &&
            (total_tokens == 1 || forceGroupedMoEVerifierPrefill(device)) &&
            use_expert_overlay &&
            overlay_plan &&
            canUseLocalTPReplicatedFastPath(*overlay_plan, device);
        const bool masked_local_tp_apportioned_decode_runtime_table =
            (local_decode_layer ||
             grouped_main_verifier_layer ||
             (mtp_sidecar_context && total_tokens == 1)) &&
            device.is_gpu() &&
            !use_expert_overlay &&
            config_.moe.routed_compute_policy == RoutedExpertComputePolicy::Apportioned &&
            config_.moe.owner_participant_count > 1 &&
            local_tp_ctx &&
            local_tp_ctx->degree() > 1;
        const bool runtime_table_eligible =
            static_full_local_expert_ownership ||
            masked_local_tp_overlay_decode_runtime_table ||
            captured_distributed_overlay_runtime_table ||
            full_local_tp_replicated_overlay_decode_runtime_table ||
            masked_local_tp_apportioned_decode_runtime_table;
        const MoERuntimeTableIdentity runtime_table_identity{
            .role = use_mtp_runtime_table
                        ? MoERuntimeTableRole::MTPDepth
                        : (current_batch_llep_transfer_candidate
                               ? MoERuntimeTableRole::CurrentBatchLLEPPrefill
                               : MoERuntimeTableRole::MainDecodeDurablePlacement),
            .mtp_depth = use_mtp_runtime_table ? mtp_depth_idx : -1,
        };
        /*
         * Every overlay table is a reader of the same request epoch. The main
         * table owns durable placement; MTP and current-batch LLEP are child
         * tables that borrow its banks through the exact same device ticket.
         * LLEP may mutate only its request-local assignment/arrival state.
         */
        const bool bind_overlay_epoch = use_expert_overlay;
        if (total_tokens == 1 &&
            rocm_env.moe_grouped_decode &&
            rocm_env.moe_device_routed_decode &&
            runtime_table_eligible)
        {
            moe_runtime_table = moeRuntimeTableForDevice(
                device,
                runtime_table_identity,
                total_tokens,
                runtime_table_layers,
                runtime_histogram_producer_role,
                bind_overlay_epoch);
        }
        else if (total_tokens > 1)
        {
            // Fixed-topology grouped prefill consumes routing tensors directly.
            // Decode-equivalent verifier routing instead uses the same
            // runtime-table router as serial decode, so build the table even
            // if an old environment still tries to disable grouped prefill;
            // otherwise the strict verifier row proof would fail closed before
            // reaching the rows under test.
            // Current-batch LLEP receives a prefill-only typed identity because
            // its transfer-backed placement must not leak into static decode.
            // The typed workload distinguishes ordinary prefill/MTP sidecars
            // from a grouped main verifier. Only the verifier is a production
            // decode-evidence source and therefore registers the host drain.
            moe_runtime_table = moeRuntimeTableForDevice(
                device,
                runtime_table_identity,
                total_tokens,
                runtime_table_layers,
                runtime_histogram_producer_role,
                bind_overlay_epoch);
        }

        /* The canonical ExpertOverlay parent is sized from the complete routed
         * weight manifest, which may include one or more NextN/MTP source
         * layers beyond the main transformer interval. Every maintenance
         * buffer, transfer-format scan, collective identity, and controller
         * bound must use that materialized table capacity. Using n_layers here
         * would let the forward graph bind a larger parent while retaining a
         * smaller maintenance transaction, making request reset fail and
         * leaving the trailing MTP placement outside the sole authority. */
        const int bound_runtime_table_layers =
            moe_runtime_table
                ? moe_runtime_table->layerCount()
                : runtime_table_layers;

        std::optional<DeviceMoETransferSlotDirectory::FormatProfile>
            graph_rebalance_transfer_profile;
        std::optional<DeviceMoERebalanceTransferMode> graph_rebalance_transfer_mode;
        auto collectGraphRebalanceTransferProfile =
            [&]() -> DeviceMoETransferSlotDirectory::FormatProfile
        {
            /*
             * One transfer directory is shared by the complete device-local MoE
             * domain. Its slots retain arrivals from different layers, so sizing
             * it from the current layer would either reject a later, wider
             * codebook or force one full directory allocation per layer. Scan
             * every compute-ready registry descriptor before any directory is
             * created and merge those formats into one immutable capacity
             * profile.
             */
            std::vector<std::vector<DeviceMoETransferSlotDirectory::ProjectionSpec>>
                layer_formats;
            layer_formats.reserve(
                static_cast<size_t>(bound_runtime_table_layers));

            auto weight_mgr =
                model_ctx_ ? model_ctx_->concreteWeightManager() : nullptr;
            if (!weight_mgr)
            {
                throw std::runtime_error(
                    "Qwen35 MoE graph requires a prepared ExpertGemmRegistry "
                    "before sizing transfer slots on " +
                    device.to_string());
            }
            const auto &registry = weight_mgr->expertGemmRegistry();

            for (int scan_layer = 0;
                 scan_layer < bound_runtime_table_layers;
                 ++scan_layer)
            {
                auto append_registered_formats =
                    [&](const std::vector<ITensorGemm *> &gate_engines,
                        const std::vector<ITensorGemm *> &up_engines,
                        const std::vector<ITensorGemm *> &down_engines,
                        const std::string &registry_scope) -> bool
                {
                    if (gate_engines.size() !=
                            static_cast<size_t>(config_.moe.num_experts) ||
                        up_engines.size() !=
                            static_cast<size_t>(config_.moe.num_experts) ||
                        down_engines.size() !=
                            static_cast<size_t>(config_.moe.num_experts))
                    {
                        return false;
                    }

                    bool found_complete_expert = false;
                    for (int expert = 0;
                         expert < config_.moe.num_experts;
                         ++expert)
                    {
                        ITensorGemm *gate_engine =
                            gate_engines[static_cast<size_t>(expert)];
                        ITensorGemm *up_engine =
                            up_engines[static_cast<size_t>(expert)];
                        ITensorGemm *down_engine =
                            down_engines[static_cast<size_t>(expert)];
                        if (!gate_engine || !up_engine || !down_engine)
                            continue;

                        auto exact_specs =
                            transferSlotSpecsFromPreparedExpertEngines(
                                gate_engine,
                                up_engine,
                                down_engine,
                                config_.d_model,
                                config_.moe.intermediate_size);
                        if (!exact_specs.has_value())
                        {
                            throw std::runtime_error(
                                "Qwen35 MoE graph found an invalid NativeVNNI "
                                "descriptor triple for model layer " +
                                std::to_string(scan_layer) + " expert " +
                                std::to_string(expert) + " in " +
                                registry_scope + " on " +
                                device.to_string());
                        }
                        layer_formats.push_back(std::move(*exact_specs));
                        found_complete_expert = true;
                    }
                    return found_complete_expert;
                };

                bool found_layer_format = false;
                if (overlay_runtime_plan)
                {
                    for (const auto &domain : overlay_runtime_plan->domains())
                    {
                        if (!domainContainsDevice(domain, device))
                            continue;

                        std::vector<ITensorGemm *> gate_engines;
                        std::vector<ITensorGemm *> up_engines;
                        std::vector<ITensorGemm *> down_engines;
                        (void)registry.populateExpertEnginesForDomain(
                            domain.name,
                            device,
                            scan_layer,
                            config_.moe.num_experts,
                            gate_engines,
                            up_engines,
                            down_engines);
                        found_layer_format =
                            append_registered_formats(
                                gate_engines,
                                up_engines,
                                down_engines,
                                "domain '" + domain.name + "'") ||
                            found_layer_format;
                    }
                }
                else
                {
                    std::vector<ITensorGemm *> gate_engines;
                    std::vector<ITensorGemm *> up_engines;
                    std::vector<ITensorGemm *> down_engines;
                    (void)registry.populateExpertEngines(
                        device,
                        scan_layer,
                        config_.moe.num_experts,
                        gate_engines,
                        up_engines,
                        down_engines);
                    found_layer_format =
                        append_registered_formats(
                            gate_engines,
                            up_engines,
                            down_engines,
                            "device registry");
                }

                if (!found_layer_format)
                {
                    throw std::runtime_error(
                        "Qwen35 MoE graph could not derive NativeVNNI "
                        "transfer-slot specs from prepared engines for model layer " +
                        std::to_string(scan_layer) + " on " + device.to_string());
                }
            }
            return DeviceMoETransferSlotDirectory::profileForLayerFormats(
                layer_formats);
        };
        auto graphRebalanceDomainKey = [&]() -> std::string
        {
            std::ostringstream key;
            key << device.to_string()
                << ":participant=" << config_.tp_device_idx
                << ":layers=" << bound_runtime_table_layers
                << ":experts=" << config_.moe.num_experts
                << ":topk=" << config_.moe.top_k;
            return key.str();
        };
        auto graphRebalanceBindingKey = [&]() -> std::string
        {
            std::string key = graphRebalanceDomainKey();
            if (prefill_llep_transfer_candidate)
            {
                /*
                 * Prefix rehydration and current-batch LLEP intentionally
                 * share the immutable transfer directory, but they are
                 * different graph transactions with different evidence and
                 * event lifetimes.  Giving each use an explicit logical key
                 * prevents whichever graph happens to build first from
                 * donating its policy or publication semantics to the other.
                 */
                switch (prefill_transfer_binding_role)
                {
                case GraphSideRebalanceBindingRole::CurrentBatchLLEPTransfer:
                    key += ":current_batch_llep_layer=";
                    break;
                case GraphSideRebalanceBindingRole::PrefixRuntimeRehydrationTransfer:
                    key += ":prefix_runtime_rehydration_layer=";
                    break;
                case GraphSideRebalanceBindingRole::DecodeMaintenance:
                    throw std::logic_error(
                        "Qwen35 MoE prefill transfer key received decode-maintenance policy");
                }
                key += std::to_string(layer_idx);
            }
            return key;
        };
        auto graphRebalanceCollectiveKey = [&]() -> std::string
        {
            std::ostringstream key;
            key << "backend=" << static_cast<int>(local_tp_ctx ? local_tp_ctx->backend() : CollectiveBackendType::AUTO)
                << ":degree=" << (local_tp_ctx ? local_tp_ctx->degree() : 0)
                << ":layers=" << bound_runtime_table_layers
                << ":experts=" << config_.moe.num_experts
                << ":topk=" << config_.moe.top_k;
            if (local_tp_ctx)
            {
                key << ":devices=";
                const auto &devices = local_tp_ctx->devices();
                for (size_t i = 0; i < devices.size(); ++i)
                {
                    if (i > 0)
                        key << ',';
                    key << devices[i].toString();
                }
            }
            return key.str();
        };
        auto graphRebalanceWorkspaceName = [&]() -> std::string
        {
            std::string workspace =
                std::string(prefill_llep_transfer_candidate ? "moe_prefill_llep_" : "moe_device_rebalance_") +
                graphRebalanceCollectiveKey();
            if (prefill_llep_transfer_candidate)
            {
                /*
                 * A transfer-backed prefill stage joins its auxiliary transfer
                 * stream back into the layer's compute stream before expert
                 * execution. Two persistent lanes therefore cover adjacent
                 * layers while allowing layer N+2 to reuse layer N's storage
                 * only after the graph's event edge has completed. Keeping the
                 * lane count fixed is essential: a payload workspace per model
                 * layer consumes several GiB for production expert shapes and
                 * starves the forward graph's activation/KV scratch budget.
                 */
                constexpr int kPrefillLLEPTransferWorkspaceLanes = 2;
                workspace += ":prefill_lane=";
                workspace += std::to_string(
                    layer_idx >= 0 ? (layer_idx % kPrefillLLEPTransferWorkspaceLanes) : 0);
            }
            return workspace;
        };
        auto makeGraphRebalanceConfig =
            [&](GraphSideRebalanceBindingRole purpose)
            -> DeviceMoERebalanceConfig
        {
            const bool current_batch_llep_policy =
                purpose ==
                GraphSideRebalanceBindingRole::CurrentBatchLLEPTransfer;
            const bool decode_maintenance_policy =
                purpose == GraphSideRebalanceBindingRole::DecodeMaintenance;
            if (current_batch_llep_policy &&
                config_.moe.routed_prefill_assignment_policy !=
                    RoutedExpertAssignmentPolicy::LeastLoadedResident)
            {
                throw std::logic_error(
                    "Qwen35 MoE current-batch LLEP binding requires the "
                    "declarative least-loaded-resident prefill policy");
            }

            auto deviceRebalanceConfigOrEnv =
                [&](uint32_t config_value, const char *env_name, int env_value) -> uint32_t
            {
                if (env.presence.has(env_name))
                    return static_cast<uint32_t>(std::max(0, env_value));
                return config_value;
            };

            DeviceMoERebalanceConfig rebalance_config;
            rebalance_config.num_layers =
                static_cast<uint32_t>(bound_runtime_table_layers);
            rebalance_config.num_experts = static_cast<uint32_t>(config_.moe.num_experts);
            rebalance_config.top_k = static_cast<uint32_t>(config_.moe.top_k);
            rebalance_config.participant_id = static_cast<uint32_t>(config_.tp_device_idx);
            rebalance_config.participant_count = static_cast<uint32_t>(local_tp_ctx ? local_tp_ctx->degree() : 0);
            rebalance_config.root_participant = static_cast<uint32_t>(
                overlay_plan ? continuationRootParticipant(*overlay_plan) : 0);
            const int current_batch_assignment_window =
                config_.moe.routed_prefill_config.assignment_window_tokens > 0
                    ? config_.moe.routed_prefill_config.assignment_window_tokens
                    : total_tokens;
            rebalance_config.window_size_tokens = static_cast<uint32_t>(
                std::max(
                    1,
                    current_batch_llep_policy
                        ? current_batch_assignment_window
                        : (decode_maintenance_policy
                               ? config_.moe.rebalance_config.window_size
                               : total_tokens)));
            const int maintenance_slack =
                !decode_maintenance_policy
                    ? 0
                    : config_.moe.rebalance_config
                                  .device_maintenance_slack_tokens >= 0
                    ? std::max(
                          0,
                          config_.moe.rebalance_config
                              .device_maintenance_slack_tokens)
                    : std::max(
                          0,
                          env.moe_rebalance
                              .device_rebalance_maintenance_slack_tokens);
            const int requested_maintenance_period =
                std::max(
                    1,
                    static_cast<int>(rebalance_config.window_size_tokens) +
                        maintenance_slack);
            const int configured_minimum_period =
                !decode_maintenance_policy
                    ? 0
                    : config_.moe.rebalance_config
                            .device_min_maintenance_period_tokens >= 0
                    ? std::max(
                          0,
                          config_.moe.rebalance_config
                              .device_min_maintenance_period_tokens)
                    : std::max(
                          0,
                          env.moe_rebalance
                              .device_rebalance_min_maintenance_period_tokens);
            const int maintenance_period =
                configured_minimum_period > 0
                    ? std::max(
                          requested_maintenance_period,
                          configured_minimum_period)
                    : requested_maintenance_period;
            const int configured_initial_period =
                !decode_maintenance_policy
                    ? 0
                    : config_.moe.rebalance_config
                            .device_initial_maintenance_period_tokens >= 0
                    ? std::max(
                          0,
                          config_.moe.rebalance_config
                              .device_initial_maintenance_period_tokens)
                    : std::max(
                          0,
                          env.moe_rebalance
                              .device_rebalance_initial_maintenance_period_tokens);
            rebalance_config.maintenance_period_tokens =
                static_cast<uint32_t>(maintenance_period);
            rebalance_config.initial_maintenance_period_tokens =
                static_cast<uint32_t>(
                    configured_initial_period > 0
                        ? configured_initial_period
                        : maintenance_period);
            rebalance_config.max_hot_replicas_per_participant = static_cast<uint32_t>(
                decode_maintenance_policy
                    ? std::min(hot_replica_cap, config_.moe.num_experts)
                    : 0);
            rebalance_config.dynamic_imbalance_threshold_per_mille =
                decode_maintenance_policy
                    ? config_.moe.rebalance_config
                          .dynamic_imbalance_threshold_per_mille
                    : 0u;
            rebalance_config.dynamic_min_improvement_per_mille =
                decode_maintenance_policy
                    ? config_.moe.rebalance_config
                          .dynamic_min_improvement_per_mille
                    : 0u;
            rebalance_config.dynamic_max_swaps_per_layer =
                decode_maintenance_policy
                    ? config_.moe.rebalance_config.dynamic_max_swaps_per_layer
                    : 0u;
            rebalance_config.dynamic_max_plan_entries_per_wave =
                decode_maintenance_policy
                    ? config_.moe.rebalance_config
                          .dynamic_max_plan_entries_per_wave
                    : 0u;
            rebalance_config.dynamic_min_window_activations =
                static_cast<uint32_t>(std::min<uint64_t>(
                    decode_maintenance_policy
                        ? config_.moe.rebalance_config
                              .dynamic_min_window_activations
                        : 0ULL,
                    static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())));
            switch (purpose)
            {
            case GraphSideRebalanceBindingRole::DecodeMaintenance:
                rebalance_config.routed_assignment_policy =
                    config_.moe.routed_decode_assignment_policy ==
                            RoutedExpertAssignmentPolicy::LeastLoadedResident
                        ? kDeviceMoERebalanceAssignmentLeastLoadedResident
                        : kDeviceMoERebalanceAssignmentStaticOwner;
                rebalance_config.min_load_spread_improvement =
                    deviceRebalanceConfigOrEnv(
                        config_.moe.rebalance_config
                            .device_min_load_spread_improvement,
                        "LLAMINAR_MOE_DEVICE_REBALANCE_MIN_LOAD_SPREAD_IMPROVEMENT",
                        env.moe_rebalance
                            .device_rebalance_min_load_spread_improvement);
                rebalance_config.min_load_spread_improvement_divisor =
                    deviceRebalanceConfigOrEnv(
                        config_.moe.rebalance_config
                            .device_min_load_spread_improvement_divisor,
                        "LLAMINAR_MOE_DEVICE_REBALANCE_MIN_LOAD_SPREAD_IMPROVEMENT_DIVISOR",
                        env.moe_rebalance
                            .device_rebalance_min_load_spread_improvement_divisor);
                rebalance_config
                    .min_wave_spread_improvement_per_payload_slot =
                    deviceRebalanceConfigOrEnv(
                        config_.moe.rebalance_config
                            .device_min_wave_spread_improvement_per_payload_slot,
                        "LLAMINAR_MOE_DEVICE_REBALANCE_MIN_WAVE_SPREAD_IMPROVEMENT_PER_PAYLOAD_SLOT",
                        env.moe_rebalance
                            .device_rebalance_min_wave_spread_improvement_per_payload_slot);
                rebalance_config
                    .min_foreign_rows_per_critical_path_payload_slot =
                    deviceRebalanceConfigOrEnv(
                        config_.moe.rebalance_config
                            .device_min_foreign_rows_per_critical_path_payload_slot,
                        "LLAMINAR_MOE_DEVICE_REBALANCE_MIN_FOREIGN_ROWS_PER_CRITICAL_PATH_PAYLOAD_SLOT",
                        env.moe_rebalance
                            .device_rebalance_min_foreign_rows_per_critical_path_payload_slot);
                break;
            case GraphSideRebalanceBindingRole::CurrentBatchLLEPTransfer:
                /*
                 * Paper-style LLEP owns only this batch's row assignment. Its
                 * alpha/lambda economics are declared by routed_prefill_config;
                 * durable residency-maintenance floors must never suppress or
                 * reshape the current-batch planner. A movement-positive test
                 * can therefore disable the lambda skip without also enabling
                 * Dynamic residency maintenance or mutating unrelated knobs.
                 */
                rebalance_config.routed_assignment_policy =
                    kDeviceMoERebalanceAssignmentLeastLoadedResident;
                rebalance_config.min_load_spread_improvement = 0u;
                rebalance_config.min_load_spread_improvement_divisor = 0u;
                rebalance_config
                    .min_wave_spread_improvement_per_payload_slot = 0u;
                rebalance_config
                    .min_foreign_rows_per_critical_path_payload_slot = 0u;
                break;
            case GraphSideRebalanceBindingRole::PrefixRuntimeRehydrationTransfer:
                /*
                 * Prefix restore rehydrates an already selected placement. It
                 * never performs a fresh least-loaded assignment and therefore
                 * carries no decode-maintenance economy thresholds.
                 */
                rebalance_config.routed_assignment_policy =
                    kDeviceMoERebalanceAssignmentStaticOwner;
                rebalance_config.min_load_spread_improvement = 0u;
                rebalance_config.min_load_spread_improvement_divisor = 0u;
                rebalance_config
                    .min_wave_spread_improvement_per_payload_slot = 0u;
                rebalance_config
                    .min_foreign_rows_per_critical_path_payload_slot = 0u;
                break;
            }
            rebalance_config.min_router_spread_improvement_per_payload_slot =
                decode_maintenance_policy
                    ? deviceRebalanceConfigOrEnv(
                          config_.moe.rebalance_config
                              .device_min_router_spread_improvement_per_payload_slot,
                          "LLAMINAR_MOE_DEVICE_REBALANCE_MIN_ROUTER_SPREAD_IMPROVEMENT_PER_PAYLOAD_SLOT",
                          env.moe_rebalance
                              .device_rebalance_min_router_spread_improvement_per_payload_slot)
                    : 0u;
            rebalance_config.max_post_wave_load_spread_per_mille =
                decode_maintenance_policy
                    ? deviceRebalanceConfigOrEnv(
                          config_.moe.rebalance_config
                              .device_max_post_wave_load_spread_per_mille,
                          "LLAMINAR_MOE_DEVICE_REBALANCE_MAX_POST_WAVE_LOAD_SPREAD_PERMILLE",
                          env.moe_rebalance
                              .device_rebalance_max_post_wave_load_spread_per_mille)
                    : 0u;
            if (current_batch_llep_policy)
            {
                rebalance_config.llep_alpha_numerator =
                    std::max<uint32_t>(
                        1u,
                        config_.moe.routed_prefill_config
                            .llep_alpha_numerator);
                rebalance_config.llep_alpha_denominator =
                    std::max<uint32_t>(
                        1u,
                        config_.moe.routed_prefill_config
                            .llep_alpha_denominator);
                rebalance_config.llep_lambda_numerator =
                    std::max<uint32_t>(
                        1u,
                        config_.moe.routed_prefill_config
                            .llep_lambda_numerator);
                rebalance_config.llep_lambda_denominator =
                    std::max<uint32_t>(
                        1u,
                        config_.moe.routed_prefill_config
                            .llep_lambda_denominator);
                rebalance_config.llep_enable_balanced_skip =
                    config_.moe.routed_prefill_config
                            .llep_enable_balanced_skip
                        ? 1u
                        : 0u;
            }
            rebalance_config.flags =
                static_cast<uint32_t>(DeviceMoERebalanceFlags::ResetHistogramsAfterApply);
            if (rebalance_config.max_hot_replicas_per_participant > 0)
            {
                rebalance_config.flags |=
                    static_cast<uint32_t>(DeviceMoERebalanceFlags::HotReplicaCache);
            }
            if (decode_maintenance_policy &&
                env.moe_rebalance.device_rebalance_maintenance_graph)
            {
                rebalance_config.flags |=
                    static_cast<uint32_t>(DeviceMoERebalanceFlags::DeferRuntimeApply);
            }
            if (env.moe_rebalance.device_rebalance_collect_load_stats ||
                PerfStatsCollector::isDomainEnabled("moe_rebalance"))
            {
                rebalance_config.flags |=
                    static_cast<uint32_t>(DeviceMoERebalanceFlags::CollectLoadStats);
            }

            if (first_device_rebalance_decode_layer &&
                rebalance_config.num_layers > 0)
            {
                const uint32_t current_layer =
                    static_cast<uint32_t>(std::max(0, layer_idx));
                if (current_layer + 1u < rebalance_config.num_layers)
                {
                    rebalance_config.layer_window_start = current_layer + 1u;
                    rebalance_config.layer_window_count =
                        rebalance_config.num_layers - rebalance_config.layer_window_start;
                }
                else if (current_layer < rebalance_config.num_layers)
                {
                    rebalance_config.layer_window_start = current_layer;
                    rebalance_config.layer_window_count = 1u;
                }
            }
            rebalance_config.layer_wave_count =
                static_cast<uint32_t>(
                    std::max(0, env.moe_rebalance.device_rebalance_layer_wave_count));
            return rebalance_config;
        };
        auto graphRebalanceMovesFixedPayloadCapacity =
            [](DeviceMoERebalanceTransferMode mode) -> bool
        {
            return deviceMoERebalanceModeMovesFixedPayloadCapacity(mode);
        };
        auto graphRebalanceTransferModeName =
            [](DeviceMoERebalanceTransferMode mode) -> const char *
        {
            switch (mode)
            {
            case DeviceMoERebalanceTransferMode::ResidentOnly:
                return "resident_only";
            case DeviceMoERebalanceTransferMode::CompactTransferSlots:
                return "compact_transfer_slots";
            case DeviceMoERebalanceTransferMode::CollectiveSidebandPayload:
                return "collective_sideband_payload";
            case DeviceMoERebalanceTransferMode::LegacyCollectiveAllGather:
                return "legacy_collective_allgather";
            }
            return "unknown";
        };
        auto graphRebalanceUsesTransferSlots = [&]() -> bool
        {
            return graph_rebalance_transfer_mode.has_value() &&
                   deviceMoERebalanceModeUsesTransferSlots(*graph_rebalance_transfer_mode);
        };
        auto graphRebalanceFixedPayloadTransferEnabled = [&]() -> bool
        {
            return graph_rebalance_transfer_mode.has_value() &&
                   graphRebalanceMovesFixedPayloadCapacity(*graph_rebalance_transfer_mode);
        };
        auto graphRebalanceEnsureTransferMode = [&]() -> bool
        {
            if (!graph_rebalance_transport_candidate ||
                !moe_runtime_table ||
                !local_tp_ctx)
            {
                return false;
            }

            if (!graph_rebalance_transfer_mode.has_value())
            {
                /*
                 * Durable same-tier skew correction exchanges paired complete
                 * expert owners even when the optional replica cache is zero.
                 * It therefore requires compact non-empty payload slots; a
                 * ResidentOnly selection would publish ownership metadata for
                 * weights that never arrived.
                 */
                const bool dynamic_ownership_transfers =
                    device_side_graph_rebalance_candidate;
                const bool routed_assignment_payload_transfers =
                    config_.moe.routed_decode_assignment_policy ==
                        RoutedExpertAssignmentPolicy::LeastLoadedResident ||
                    current_batch_llep_transfer_candidate;
                graph_rebalance_transfer_mode =
                    selectGraphRebalanceTransferMode(
                        *local_tp_ctx,
                        device,
                        static_cast<uint32_t>(
                            std::min(hot_replica_cap, config_.moe.num_experts)),
                        dynamic_ownership_transfers || routed_assignment_payload_transfers);
            }
            if (!graph_rebalance_transfer_mode.has_value())
            {
                throw std::runtime_error(
                    "Qwen35 MoE graph-side rebalance requires graph-captured NCCL/RCCL maintenance-wave transport for " +
                    device.to_string() +
                    "; refusing to fall back to host publish/apply in device-side mode");
            }

            return true;
        };
        auto graphRebalanceDecodeUsesMutableDescriptors = [&]() -> bool
        {
            if (!device_side_graph_rebalance_candidate)
                return false;
            if (!graphRebalanceEnsureTransferMode())
                return false;

            /*
             * ResidentOnly changes runtime top-k/local-compute masks for
             * experts that already have local resident descriptors, so the
             * immutable grouped descriptor tables remain valid. Compact and
             * fixed-payload transfer-slot modes can publish newly-arrived
             * descriptors and must therefore use mutable runtime descriptors.
             */
            return deviceMoERebalanceModeUsesTransferSlots(*graph_rebalance_transfer_mode);
        };
        auto activeRuntimeBankUsesTransientLocalPayload = [&](int table_layer_idx) -> bool
        {
            if (!moe_runtime_table || table_layer_idx < 0)
                return false;
            if (moe_runtime_table->decodeRuntimePublicationRequired(table_layer_idx))
                return false;
            return deviceMoELayerUsesTransientLocalPayload(
                moe_runtime_table->hostLayerState(table_layer_idx));
        };
        auto ensureGraphRebalanceTransferMode = [&]() -> bool
        {
            if (!graph_rebalance_transport_candidate ||
                !moe_runtime_table ||
                !local_tp_ctx)
            {
                return false;
            }

            return graphRebalanceEnsureTransferMode();
        };
        auto shouldCollectGraphRebalanceTransferSpecs = [&]() -> bool
        {
            if (!ensureGraphRebalanceTransferMode())
                return false;
            return graphRebalanceUsesTransferSlots();
        };
        auto graphRebalanceTransferDirectoryKey =
            [&](uint32_t transfer_slot_count) -> std::string
        {
            if (!graph_rebalance_transfer_profile.has_value())
            {
                throw std::logic_error(
                    "Qwen35 MoE cannot identify a transfer directory without "
                    "a NativeVNNI format profile");
            }

            /*
             * This key identifies physical storage, not only a logical TP
             * domain. graphRebalanceDomainKey() already includes the exact
             * DeviceId and participant. Keeping all remaining allocation
             * discriminators in this one builder prevents prefill and decode
             * maintenance from silently inventing different cache identities.
             */
            std::ostringstream key;
            key << graphRebalanceDomainKey()
                << ":slots=" << transfer_slot_count;
            for (const auto &spec :
                 graph_rebalance_transfer_profile->allocation_specs)
            {
                key << ':' << spec.label
                    << '=' << spec.N << 'x' << spec.K
                    << ":cb" << static_cast<int>(spec.codebook_id)
                    << ":pb" << spec.payload_bytes_per_block
                    << ":asym" << (spec.is_asymmetric ? 1 : 0)
                    << ":emins" << (spec.has_emins ? 1 : 0);
            }
            key << ":wire="
                << graph_rebalance_transfer_profile->max_wire_payload_bytes;
            return key.str();
        };
        auto getOrCreateGraphRebalanceTransferDirectory =
            [&](const DeviceMoERebalanceConfig &rebalance_config,
                DeviceMoETransferSlotDirectory::BufferedCapacity
                    transfer_capacity,
                const char *context)
            -> std::pair<
                std::string,
                std::shared_ptr<DeviceMoETransferSlotDirectory>>
        {
            if (!graph_rebalance_transfer_profile.has_value())
            {
                throw std::logic_error(
                    "Qwen35 MoE transfer-directory creation requires a "
                    "NativeVNNI format profile");
            }

            const uint32_t transfer_slot_count =
                transfer_capacity.total_slots;

            IBackend *backend = getBackendFor(device);
            const int gpu_ordinal = gpuOrdinalForGraphDevice(device);
            if (!backend || gpu_ordinal < 0)
            {
                throw std::runtime_error(
                    "Qwen35 MoE could not resolve the transfer-directory "
                    "backend for " +
                    device.to_string() +
                    (context ? std::string(" (") + context + ")"
                             : std::string{}));
            }

            const std::string transfer_key =
                graphRebalanceTransferDirectoryKey(transfer_slot_count);
            auto existing =
                moe_transfer_slot_directories_.find(transfer_key);
            if (existing == moe_transfer_slot_directories_.end())
            {
                if (!prepared_weight_store_)
                {
                    throw std::logic_error(
                        "Qwen35 MoE transfer-directory construction has no model-owned PreparedWeightStore");
                }
                const auto memory_authority =
                    prepared_weight_store_->physicalMemoryAuthority();
                if (!memory_authority)
                {
                    throw std::logic_error(
                        "Qwen35 MoE transfer-directory construction reached allocation before physical-memory admission");
                }
                auto directory = DeviceMoETransferSlotDirectory::create(
                    backend,
                    device,
                    gpu_ordinal,
                    rebalance_config.participant_id,
                    transfer_capacity,
                    *graph_rebalance_transfer_profile,
                    memory_authority);
                existing =
                    moe_transfer_slot_directories_
                        .emplace(transfer_key, std::move(directory))
                        .first;
            }
            if (!existing->second)
            {
                throw std::logic_error(
                    "Qwen35 MoE transfer-directory cache contains a null "
                    "physical owner for " +
                    transfer_key);
            }

            /*
             * A permissive peer-memory topology can make an ownership mistake
             * numerically invisible while turning each expert GEMV into
             * fine-grained PCIe traffic. Treat reuse across a physical owner
             * boundary as a fatal graph-construction error.
             */
            existing->second->requirePhysicalOwner(
                device,
                gpu_ordinal,
                rebalance_config.participant_id);
            return {transfer_key, existing->second};
        };
        auto ensureGraphRebalanceTransferBindingOnly =
            [&](const char *context) -> const GraphSideRebalanceBinding *
        {
            if (!graph_rebalance_transport_candidate ||
                !moe_runtime_table ||
                !local_tp_ctx)
            {
                return nullptr;
            }

            const std::string binding_key = graphRebalanceBindingKey();
            DeviceMoERebalanceConfig rebalance_config =
                makeGraphRebalanceConfig(prefill_transfer_binding_role);
            if (!validateDeviceMoERebalanceConfig(rebalance_config))
                return nullptr;
            if (!ensureGraphRebalanceTransferMode())
                return nullptr;
            if (!graphRebalanceUsesTransferSlots())
                return nullptr;
            if (prefill_llep_transfer_candidate &&
                graph_rebalance_transfer_mode.value() !=
                    DeviceMoERebalanceTransferMode::CompactTransferSlots)
            {
                throw std::runtime_error(
                    "Qwen35 MoE LLEP prefill requires compact transfer-slot payload movement for " +
                    device.to_string() + "; refusing fixed-payload/host fallback");
            }

            const auto existing =
                moe_graph_rebalance_bindings_.find(binding_key);
            if (existing != moe_graph_rebalance_bindings_.end())
            {
                const auto &binding = existing->second;
                if (binding.role != prefill_transfer_binding_role ||
                    binding.device_id != device ||
                    binding.collective_tp_ctx != local_tp_ctx ||
                    binding.moe_runtime_table != moe_runtime_table ||
                    binding.tp_device_idx != config_.tp_device_idx ||
                    binding.transfer_mode !=
                        graph_rebalance_transfer_mode.value())
                {
                    throw std::logic_error(
                        "Qwen35 MoE prefill transfer binding identity changed "
                        "while reusing " +
                        binding_key);
                }
                return &binding;
            }

            if (deviceMoERebalanceModePlansMissingArrivals(
                    graph_rebalance_transfer_mode.value()))
            {
                rebalance_config.flags |=
                    static_cast<uint32_t>(DeviceMoERebalanceFlags::PlanMissingArrivals);
            }
            if (!graph_rebalance_transfer_profile.has_value())
            {
                throw std::runtime_error(
                    "Qwen35 MoE graph-side transfer binding requires a NativeVNNI transfer profile for layer " +
                    std::to_string(layer_idx) + " on " + device.to_string() +
                    (context ? std::string(" (") + context + ")" : std::string{}));
            }

            const uint64_t prefill_llep_slots =
                prefill_llep_transfer_candidate
                    ? static_cast<uint64_t>(
                          std::max(1, env.moe_rebalance.device_rebalance_compact_payload_slots))
                    : 1ULL;
            const auto transfer_capacity =
                DeviceMoETransferSlotDirectory::planRuntimeCapacity(
                    rebalance_config,
                    prefill_llep_slots,
                    static_cast<uint32_t>(
                        std::max(1, env.moe_rebalance.gpu_direct_transfer_wave_experts)),
                    static_cast<uint32_t>(
                        std::max(1, env.moe_rebalance.gpu_direct_transfer_buffers)));
            rebalance_config.active_transfer_slot_capacity =
                transfer_capacity.active_slots;
            rebalance_config.transfer_slot_directory_capacity =
                transfer_capacity.total_slots;
            auto [transfer_key, transfer_directory] =
                getOrCreateGraphRebalanceTransferDirectory(
                    rebalance_config,
                    transfer_capacity,
                    context);

            const std::string rebalance_workspace = graphRebalanceWorkspaceName();
            /*
             * Transfer payloads and event edges are layer-owned; only the
             * auxiliary stream and workspace storage use the bounded rolling
             * lane.  A CUDA/HIP graph may contain many records and waits on the
             * same auxiliary stream, but reusing one event object for layer 0,
             * layer 2, layer 4, ... makes those graph dependencies alias the
             * event's latest record.  That ambiguity used to invalidate
             * long-context capture after several MoE layers.
             *
             * Key the state object by the layer-specific transfer identity so
             * every stage owns a persistent compute-ready/transfer-done event
             * pair. DeviceMoERebalanceTransferState::ensure() still asks the
             * worker context for the auxiliary stream by rebalance_workspace,
             * so all even layers share lane 0 and all odd layers share lane 1.
             * The result is two streams and two workspace lanes, with explicit
             * non-aliasing graph edges for every layer.
             */
            const std::string transfer_state_key =
                binding_key + ":workspace=" + rebalance_workspace;
            auto &state_ref =
                moe_rebalance_transfer_states_[transfer_state_key];
            if (!state_ref)
                state_ref = std::make_shared<DeviceMoERebalanceTransferState>();

            const uint32_t collective_payload_slot_capacity =
                std::min<uint32_t>(
                    transfer_directory->slotCount(),
                    static_cast<uint32_t>(
                        std::max(1, env.moe_rebalance.device_rebalance_compact_payload_slots)));
            const uint64_t collective_payload_slot_bytes =
                DeviceMoERebalanceWorkspaceContract::
                    collectivePayloadSlotBytes(
                        transfer_directory->wirePayloadBytes());

            moe_graph_rebalance_bindings_[binding_key] = GraphSideRebalanceBinding{
                .transfer_key = transfer_key,
                .workspace_name = rebalance_workspace,
                .role = prefill_transfer_binding_role,
                .device_id = device,
                .collective_tp_ctx = local_tp_ctx,
                .moe_runtime_table = moe_runtime_table,
                .tp_device_idx = config_.tp_device_idx,
                .config = rebalance_config,
                .local_transfer_slots = transfer_directory->deviceEntries(),
                .local_transfer_slot_count = transfer_directory->slotCount(),
                .transfer_mode = graph_rebalance_transfer_mode.value(),
                .collective_payload_slot_bytes = collective_payload_slot_bytes,
                .collective_payload_slot_capacity = collective_payload_slot_capacity,
                .transfer_state = state_ref,
                .producer_layer_idx = layer_idx};

            return &moe_graph_rebalance_bindings_.find(binding_key)->second;
        };
        auto attachPrefillLLEPTransferBinding =
            [&](MoEExpertComputeStage::Params &expert_params,
                const char *context)
        {
            if (!prefill_llep_transfer_candidate)
                return;
            graph_rebalance_transfer_profile =
                collectGraphRebalanceTransferProfile();
            const auto *binding =
                ensureGraphRebalanceTransferBindingOnly(context);
            if (!binding)
            {
                throw std::runtime_error(
                    "Qwen35 MoE LLEP prefill could not create transfer binding for layer " +
                    std::to_string(layer_idx) + " on " + device.to_string());
            }
            expert_params.prefill_llep_tp_ctx = binding->collective_tp_ctx;
            expert_params.prefill_llep_transfer_slots =
                binding->local_transfer_slots;
            expert_params.prefill_llep_transfer_slot_count =
                binding->local_transfer_slot_count;
            expert_params.prefill_llep_payload_slot_bytes =
                binding->collective_payload_slot_bytes;
            expert_params.prefill_llep_payload_slot_capacity =
                binding->collective_payload_slot_capacity;
            expert_params.prefill_llep_transfer_mode =
                binding->transfer_mode;
            expert_params.prefix_runtime_device_rehydration =
                prefix_runtime_device_rehydration;
            if (prefix_runtime_device_rehydration)
            {
                expert_params.prefix_runtime_rehydration_transfer_state =
                    std::make_shared<DeviceMoERebalanceTransferState>();
            }
            /*
             * The binding owns persistent physical resources.  Window size,
             * maintenance period, and other policy fields belong to this
             * exact graph invocation and may vary with its prefill bucket.
             * Rebuild those fields from the declarative graph policy, then
             * project only the proven physical capacities from the binding.
             */
            DeviceMoERebalanceConfig invocation_config =
                makeGraphRebalanceConfig(prefill_transfer_binding_role);
            invocation_config.active_transfer_slot_capacity =
                binding->config.active_transfer_slot_capacity;
            invocation_config.transfer_slot_directory_capacity =
                binding->config.transfer_slot_directory_capacity;
            if (deviceMoERebalanceModePlansMissingArrivals(
                    binding->transfer_mode))
            {
                invocation_config.flags |= static_cast<uint32_t>(
                    DeviceMoERebalanceFlags::PlanMissingArrivals);
            }
            if (!validateDeviceMoERebalanceConfig(invocation_config))
            {
                throw std::logic_error(
                    "Qwen35 MoE prefill transfer invocation produced an "
                    "invalid graph-owned rebalance policy for layer " +
                    std::to_string(layer_idx) + " on " +
                    device.to_string());
            }
            expert_params.prefill_llep_rebalance_config = invocation_config;
            expert_params.prefill_llep_transfer_state =
                binding->transfer_state;
            expert_params.prefill_llep_workspace_name =
                binding->workspace_name;
        };

        bool graph_rebalance_plan_inserted = false;
        std::string graph_rebalance_apply_node;
        std::string graph_rebalance_collect_node;
        std::string graph_rebalance_plan_after_sideband_node;
        std::optional<MoEDeviceRebalanceStage::Params> graph_rebalance_plan_after_sideband_params;
        std::string graph_rebalance_pack_payload_node;
        std::optional<MoEDeviceRebalanceStage::Params> graph_rebalance_pack_payload_params;
        std::string graph_rebalance_unpack_payload_node;
        std::optional<MoEDeviceRebalanceStage::Params> graph_rebalance_unpack_payload_params;
        bool graph_rebalance_transfer_command_sideband_taken = false;
        bool graph_rebalance_transfer_payload_sideband_taken = false;
        std::string current_batch_llep_plan_node;
        std::string current_batch_llep_apply_node;
        std::string current_batch_llep_expert_node;
        std::optional<MoEGPUCurrentBatchLLEPStage::Params>
            current_batch_llep_plan_params;
        std::optional<MoEGPUCurrentBatchLLEPStage::Params>
            current_batch_llep_apply_params;
        bool current_batch_llep_payload_sideband_taken = false;
        const bool graph_rebalance_can_have_allreduce_anchor =
            config_.tp_ctx && config_.tp_ctx->degree() > 1;

        /**
         * Build the one payload sideband consumed by the shared-expert anchor.
         *
         * PlanAndPack declares and writes the send buffer. The matching apply
         * phase declares the gathered buffer. Returning this binding exactly
         * once makes duplicate collective attachment unrepresentable while
         * leaving the physical pointer resolution to TPAllreduceStage.
         */
        auto takeCurrentBatchLLEPPayloadSideband =
            [&]() -> std::vector<TPAllreduceSidebandWorkspaceBinding>
        {
            if (!current_batch_llep_plan_params.has_value() ||
                !current_batch_llep_apply_params.has_value() ||
                current_batch_llep_payload_sideband_taken)
            {
                return {};
            }
            const auto &params = *current_batch_llep_plan_params;
            const size_t payload_bytes =
                MoEGPUCurrentBatchLLEPStage::localPayloadBytes(params);
            if (payload_bytes == 0)
            {
                throw std::logic_error(
                    "Qwen35 MoE current-batch LLEP sideband has zero payload capacity for layer " +
                    std::to_string(layer_idx));
            }

            TPAllreduceSidebandWorkspaceBinding payload;
            payload.kind = LocalTPCollectiveSidebandKind::Allgather;
            payload.send_buffer_name =
                MoEGPUCurrentBatchLLEPStage::workspaceBufferName(
                    MoEDeviceRebalanceStage::WS_LOCAL_TRANSFER_PAYLOAD,
                    params.workspace_name);
            payload.recv_buffer_name =
                MoEGPUCurrentBatchLLEPStage::workspaceBufferName(
                    MoEDeviceRebalanceStage::WS_GATHERED_TRANSFER_PAYLOAD,
                    params.workspace_name);
            payload.element_count = payload_bytes;
            payload.dtype = CollectiveDataType::INT8;
            payload.root_device_index =
                static_cast<int>(params.config.root_participant);
            payload.name = "moe_current_batch_llep_payload";
            current_batch_llep_payload_sideband_taken = true;
            return {std::move(payload)};
        };
        auto makeGraphRebalanceStateSidebands =
            [&](const GraphSideRebalanceBinding &binding)
            -> std::vector<TPAllreduceSidebandWorkspaceBinding>
        {
            std::vector<TPAllreduceSidebandWorkspaceBinding> sidebands;
            if (!binding.collective_tp_ctx ||
                !binding.collective_tp_ctx->supportsCollectiveSidebandOnStreamGraphCapture())
            {
                return sidebands;
            }

            const uint32_t window_count =
                binding.config.layer_window_count == 0u
                    ? binding.config.num_layers
                    : std::min(binding.config.layer_window_count,
                               binding.config.num_layers);
            const uint32_t wave_count =
                binding.config.layer_wave_count == 0u
                    ? window_count
                    : std::min(binding.config.layer_wave_count, window_count);
            const size_t local_histogram_entries =
                static_cast<size_t>(std::max<uint32_t>(1u, wave_count)) *
                static_cast<size_t>(binding.config.num_experts);
            if (local_histogram_entries == 0)
                return sidebands;

            TPAllreduceSidebandWorkspaceBinding histogram_sideband;
            histogram_sideband.kind = LocalTPCollectiveSidebandKind::Allgather;
            histogram_sideband.send_buffer_name =
                MoEDeviceRebalanceStage::workspaceBufferName(
                    MoEDeviceRebalanceStage::WS_LOCAL_HISTOGRAM,
                    binding.workspace_name);
            histogram_sideband.recv_buffer_name =
                MoEDeviceRebalanceStage::workspaceBufferName(
                    MoEDeviceRebalanceStage::WS_GATHERED_HISTOGRAM,
                    binding.workspace_name);
            static_assert(sizeof(uint64_t) == 2 * sizeof(int32_t));
            histogram_sideband.element_count = local_histogram_entries * 2u;
            histogram_sideband.dtype = CollectiveDataType::INT32;
            histogram_sideband.root_device_index =
                static_cast<int>(binding.config.root_participant);
            histogram_sideband.name = "moe_rebalance_histogram_sideband";
            sidebands.push_back(std::move(histogram_sideband));

            return sidebands;
        };
        auto takeGraphRebalanceStateSidebands =
            [&]() -> std::vector<TPAllreduceSidebandWorkspaceBinding>
        {
            if (graph_rebalance_collect_node.empty())
                return {};

            const auto binding_it = moe_graph_rebalance_bindings_.find(graphRebalanceDomainKey());
            if (binding_it == moe_graph_rebalance_bindings_.end())
                return {};

            auto sidebands = makeGraphRebalanceStateSidebands(binding_it->second);
            if (!sidebands.empty())
                binding_it->second.state_sideband_enabled = true;
            return sidebands;
        };
        auto transferPlanCapacityForBinding =
            [](const GraphSideRebalanceBinding &binding) -> size_t
        {
            return static_cast<size_t>(
                deviceMoERebalanceCommandPlanCapacity(
                    binding.config,
                    binding.transfer_mode));
        };
        auto commandBufferCountForBinding =
            [](const GraphSideRebalanceBinding &binding) -> size_t
        {
            return binding.local_transfer_slot_count > 0 ? 2u : 1u;
        };
        auto makeRebalanceStageParamsFromBinding =
            [&](const GraphSideRebalanceBinding &binding,
                const std::string &stage_name,
                DeviceMoERebalanceStagePhase phase)
            -> MoEDeviceRebalanceStage::Params
        {
            MoEDeviceRebalanceStage::Params params;
            params.device_id = device;
            params.tp_ctx = binding.collective_tp_ctx;
            params.moe_runtime_table = binding.moe_runtime_table;
            params.tp_device_idx = config_.tp_device_idx;
            params.config = binding.config;
            params.local_transfer_slots = binding.local_transfer_slots;
            params.local_transfer_slot_count = binding.local_transfer_slot_count;
            params.collective_payload_slot_bytes = binding.collective_payload_slot_bytes;
            params.collective_payload_slot_capacity = binding.collective_payload_slot_capacity;
            params.stage_name = stage_name;
            params.workspace_name = binding.workspace_name;
            params.phase = phase;
            params.transfer_mode = binding.transfer_mode;
            params.transfer_state = binding.transfer_state;
            params.join_transfer_stream_after_copy = false;
            return params;
        };
        auto takeGraphRebalanceTransferSidebands =
            [&]() -> std::vector<TPAllreduceSidebandWorkspaceBinding>
        {
            const auto binding_it = moe_graph_rebalance_bindings_.find(graphRebalanceDomainKey());
            if (binding_it == moe_graph_rebalance_bindings_.end())
                return {};
            const auto &binding = binding_it->second;
            if (!binding.collective_tp_ctx ||
                !binding.collective_tp_ctx->supportsCollectiveSidebandOnStreamGraphCapture() ||
                binding.transfer_mode != DeviceMoERebalanceTransferMode::CollectiveSidebandPayload ||
                binding.local_transfer_slot_count == 0 ||
                binding.collective_payload_slot_bytes == 0 ||
                binding.producer_layer_idx < 0)
            {
                return {};
            }

            const int layer_delta = layer_idx - binding.producer_layer_idx;
            const size_t plan_capacity = transferPlanCapacityForBinding(binding);
            const size_t command_buffer_count = commandBufferCountForBinding(binding);
            if (plan_capacity == 0)
                return {};

            std::vector<TPAllreduceSidebandWorkspaceBinding> sidebands;
            bool consumed_sideband = false;
            if (layer_delta == 1 && !graph_rebalance_transfer_command_sideband_taken)
            {
                static_assert((sizeof(DeviceMoERebalancePlanEntry) % sizeof(int32_t)) == 0);
                static_assert((sizeof(DeviceMoERebalanceCommandBufferHeader) % sizeof(int32_t)) == 0);
                static_assert((sizeof(DeviceMoERebalanceWaveState) % sizeof(int32_t)) == 0);

                TPAllreduceSidebandWorkspaceBinding plan_sideband;
                plan_sideband.kind = LocalTPCollectiveSidebandKind::Allgather;
                plan_sideband.send_buffer_name =
                    MoEDeviceRebalanceStage::workspaceBufferName(
                        MoEDeviceRebalanceStage::WS_TRANSFER_PLAN,
                        binding.workspace_name);
                plan_sideband.recv_buffer_name =
                    MoEDeviceRebalanceStage::workspaceBufferName(
                        MoEDeviceRebalanceStage::WS_GATHERED_TRANSFER_PLAN,
                        binding.workspace_name);
                plan_sideband.element_count =
                    (command_buffer_count * plan_capacity *
                     sizeof(DeviceMoERebalancePlanEntry)) /
                    sizeof(int32_t);
                plan_sideband.dtype = CollectiveDataType::INT32;
                plan_sideband.root_device_index =
                    static_cast<int>(binding.config.root_participant);
                plan_sideband.name = "moe_rebalance_transfer_plan_sideband";
                sidebands.push_back(std::move(plan_sideband));

                TPAllreduceSidebandWorkspaceBinding header_sideband;
                header_sideband.kind = LocalTPCollectiveSidebandKind::Allgather;
                header_sideband.send_buffer_name =
                    MoEDeviceRebalanceStage::workspaceBufferName(
                        MoEDeviceRebalanceStage::WS_COMMAND_HEADER,
                        binding.workspace_name);
                header_sideband.recv_buffer_name =
                    MoEDeviceRebalanceStage::workspaceBufferName(
                        MoEDeviceRebalanceStage::WS_GATHERED_COMMAND_HEADER,
                        binding.workspace_name);
                header_sideband.element_count =
                    (command_buffer_count *
                     sizeof(DeviceMoERebalanceCommandBufferHeader)) /
                    sizeof(int32_t);
                header_sideband.dtype = CollectiveDataType::INT32;
                header_sideband.root_device_index =
                    static_cast<int>(binding.config.root_participant);
                header_sideband.name = "moe_rebalance_command_header_sideband";
                sidebands.push_back(std::move(header_sideband));

                TPAllreduceSidebandWorkspaceBinding wave_sideband;
                wave_sideband.kind = LocalTPCollectiveSidebandKind::Allgather;
                wave_sideband.send_buffer_name =
                    MoEDeviceRebalanceStage::workspaceBufferName(
                        MoEDeviceRebalanceStage::WS_WAVE_STATE,
                        binding.workspace_name);
                wave_sideband.recv_buffer_name =
                    MoEDeviceRebalanceStage::workspaceBufferName(
                        MoEDeviceRebalanceStage::WS_GATHERED_WAVE_STATE,
                        binding.workspace_name);
                wave_sideband.element_count =
                    (command_buffer_count *
                     sizeof(DeviceMoERebalanceWaveState)) /
                    sizeof(int32_t);
                wave_sideband.dtype = CollectiveDataType::INT32;
                wave_sideband.root_device_index =
                    static_cast<int>(binding.config.root_participant);
                wave_sideband.name = "moe_rebalance_wave_state_sideband";
                sidebands.push_back(std::move(wave_sideband));

                graph_rebalance_pack_payload_node =
                    prefix + "moe_device_rebalance_pack_payload_after_command_sideband";
                graph_rebalance_pack_payload_params =
                    makeRebalanceStageParamsFromBinding(
                        binding,
                        graph_rebalance_pack_payload_node,
                        DeviceMoERebalanceStagePhase::PackCollectivePayloadAfterSideband);
                graph_rebalance_transfer_command_sideband_taken = true;
                consumed_sideband = true;
            }
            else if (layer_delta == 2 && !graph_rebalance_transfer_payload_sideband_taken)
            {
                const size_t payload_slot_capacity =
                    std::min<size_t>(
                        plan_capacity,
                        binding.collective_payload_slot_capacity == 0
                            ? static_cast<size_t>(binding.local_transfer_slot_count)
                            : std::min<size_t>(
                                  static_cast<size_t>(binding.collective_payload_slot_capacity),
                                  static_cast<size_t>(binding.local_transfer_slot_count)));
                const size_t payload_local_bytes =
                    payload_slot_capacity *
                    (binding.transfer_mode == DeviceMoERebalanceTransferMode::CompactTransferSlots
                         ? 1u
                         : static_cast<size_t>(binding.config.participant_count)) *
                    static_cast<size_t>(binding.collective_payload_slot_bytes);
                if (payload_local_bytes == 0)
                    return {};

                TPAllreduceSidebandWorkspaceBinding payload_sideband;
                payload_sideband.kind = LocalTPCollectiveSidebandKind::Allgather;
                payload_sideband.send_buffer_name =
                    MoEDeviceRebalanceStage::workspaceBufferName(
                        MoEDeviceRebalanceStage::WS_LOCAL_TRANSFER_PAYLOAD,
                        binding.workspace_name);
                payload_sideband.recv_buffer_name =
                    MoEDeviceRebalanceStage::workspaceBufferName(
                        MoEDeviceRebalanceStage::WS_GATHERED_TRANSFER_PAYLOAD,
                        binding.workspace_name);
                payload_sideband.element_count = payload_local_bytes;
                payload_sideband.dtype = CollectiveDataType::INT8;
                payload_sideband.root_device_index =
                    static_cast<int>(binding.config.root_participant);
                payload_sideband.name = "moe_rebalance_transfer_payload_sideband";
                sidebands.push_back(std::move(payload_sideband));

                graph_rebalance_unpack_payload_node =
                    prefix + "moe_device_rebalance_unpack_payload_after_payload_sideband";
                graph_rebalance_unpack_payload_params =
                    makeRebalanceStageParamsFromBinding(
                        binding,
                        graph_rebalance_unpack_payload_node,
                        DeviceMoERebalanceStagePhase::UnpackCollectivePayloadAfterSideband);
                graph_rebalance_transfer_payload_sideband_taken = true;
                consumed_sideband = true;
            }

            if (sidebands.empty() && consumed_sideband)
            {
                graph_rebalance_transfer_command_sideband_taken = false;
                graph_rebalance_transfer_payload_sideband_taken = false;
            }
            return sidebands;
        };
        auto appendRebalanceSidebands =
            [&](std::vector<TPAllreduceSidebandWorkspaceBinding> &sidebands,
                std::vector<TPAllreduceSidebandWorkspaceBinding> extra)
        {
            for (auto &sideband : extra)
                sidebands.push_back(std::move(sideband));
        };
        auto takeGraphRebalanceSidebandsForAllreduce =
            [&]() -> std::vector<TPAllreduceSidebandWorkspaceBinding>
        {
            std::vector<TPAllreduceSidebandWorkspaceBinding> sidebands;
            if (!graph_rebalance_collect_node.empty())
                sidebands = takeGraphRebalanceStateSidebands();
            appendRebalanceSidebands(sidebands, takeGraphRebalanceTransferSidebands());
            return sidebands;
        };
        auto maybeAddGraphRebalancePayloadStageAfterSideband =
            [&](const std::string &anchor_name,
                std::string &terminal_name)
        {
            if (graph_rebalance_pack_payload_params.has_value() &&
                !graph_rebalance_pack_payload_node.empty())
            {
                graph.addNode(
                    graph_rebalance_pack_payload_node,
                    ComputeStageFactory::createMoEDeviceRebalance(
                        *graph_rebalance_pack_payload_params),
                    device);
                graph.addDependency(graph_rebalance_pack_payload_node, anchor_name);
                terminal_name = graph_rebalance_pack_payload_node;
                graph_rebalance_pack_payload_node.clear();
                graph_rebalance_pack_payload_params.reset();
            }
            if (graph_rebalance_unpack_payload_params.has_value() &&
                !graph_rebalance_unpack_payload_node.empty())
            {
                graph.addNode(
                    graph_rebalance_unpack_payload_node,
                    ComputeStageFactory::createMoEDeviceRebalance(
                        *graph_rebalance_unpack_payload_params),
                    device);
                graph.addDependency(graph_rebalance_unpack_payload_node, anchor_name);
                terminal_name = graph_rebalance_unpack_payload_node;
                graph_rebalance_unpack_payload_node.clear();
                graph_rebalance_unpack_payload_params.reset();
            }
        };
        auto maybeInsertGraphSideRebalance =
            [&](const char *context,
                const std::string &graph_rebalance_producer_node_name) -> std::string
        {
            if (!device_side_graph_rebalance_candidate ||
                !moe_runtime_table ||
                !local_tp_ctx)
            {
                return {};
            }

            if (!graph_rebalance_apply_node.empty())
                return graph_rebalance_apply_node;

            DeviceMoERebalanceConfig rebalance_config =
                makeGraphRebalanceConfig(
                    GraphSideRebalanceBindingRole::DecodeMaintenance);
            if (!validateDeviceMoERebalanceConfig(rebalance_config))
            {
                return {};
            }

            const std::string domain_key = graphRebalanceDomainKey();
            if (first_device_rebalance_decode_layer &&
                !graph_rebalance_plan_inserted)
            {
                moe_graph_rebalance_bindings_.erase(domain_key);
                if (!ensureGraphRebalanceTransferMode())
                    return {};

                const bool transfer_slot_mode_enabled =
                    graphRebalanceUsesTransferSlots();
                const std::string rebalance_workspace =
                    "moe_device_rebalance_" + graphRebalanceCollectiveKey();
                std::string transfer_key = domain_key + ":resident_only";
                DeviceMoEExpertDirectoryEntry *local_transfer_slots = nullptr;
                uint32_t local_transfer_slot_count = 0;
                uint64_t collective_payload_slot_bytes = 0;
                uint32_t collective_payload_slot_capacity = 0;
                std::shared_ptr<DeviceMoERebalanceTransferState> transfer_state;

                if (transfer_slot_mode_enabled)
                {
                    if (deviceMoERebalanceModePlansMissingArrivals(
                            graph_rebalance_transfer_mode.value()))
                    {
                        rebalance_config.flags |=
                            static_cast<uint32_t>(DeviceMoERebalanceFlags::PlanMissingArrivals);
                    }
                    if (!graph_rebalance_transfer_profile.has_value())
                    {
                        throw std::runtime_error(
                            "Qwen35 MoE graph-side rebalance requires a NativeVNNI transfer profile for layer " +
                            std::to_string(layer_idx) + " on " + device.to_string() +
                            (context ? std::string(" (") + context + ")" : std::string{}));
                    }

                    const auto transfer_capacity =
                        DeviceMoETransferSlotDirectory::planRuntimeCapacity(
                            rebalance_config,
                            /*minimum_active_slots=*/0u,
                            static_cast<uint32_t>(
                                std::max(
                                    1,
                                    env.moe_rebalance.gpu_direct_transfer_wave_experts)),
                            static_cast<uint32_t>(
                                std::max(
                                    1,
                                    env.moe_rebalance.gpu_direct_transfer_buffers)));
                    rebalance_config.active_transfer_slot_capacity =
                        transfer_capacity.active_slots;
                    rebalance_config.transfer_slot_directory_capacity =
                        transfer_capacity.total_slots;
                    auto directory_result =
                        getOrCreateGraphRebalanceTransferDirectory(
                            rebalance_config,
                            transfer_capacity,
                            context);
                    transfer_key = std::move(directory_result.first);
                    auto transfer_directory =
                        std::move(directory_result.second);

                    local_transfer_slots = transfer_directory->deviceEntries();
                    local_transfer_slot_count = transfer_directory->slotCount();
                    collective_payload_slot_capacity =
                        std::min<uint32_t>(
                            local_transfer_slot_count,
                            static_cast<uint32_t>(
                                std::max(1, env.moe_rebalance.device_rebalance_compact_payload_slots)));
                    auto &state_ref =
                        moe_rebalance_transfer_states_[transfer_key + ":workspace=" + rebalance_workspace];
                    if (!state_ref)
                        state_ref = std::make_shared<DeviceMoERebalanceTransferState>();
                    transfer_state = state_ref;
                    collective_payload_slot_bytes =
                        DeviceMoERebalanceWorkspaceContract::
                            collectivePayloadSlotBytes(
                                transfer_directory->wirePayloadBytes());
                }

                moe_graph_rebalance_bindings_[domain_key] = GraphSideRebalanceBinding{
                    .transfer_key = transfer_key,
                    .workspace_name = rebalance_workspace,
                    .role = GraphSideRebalanceBindingRole::DecodeMaintenance,
                    .device_id = device,
                    .collective_tp_ctx = local_tp_ctx,
                    .moe_runtime_table = moe_runtime_table,
                    .tp_device_idx = config_.tp_device_idx,
                    .config = rebalance_config,
                    .local_transfer_slots = local_transfer_slots,
                    .local_transfer_slot_count = local_transfer_slot_count,
                    .transfer_mode = graph_rebalance_transfer_mode.value(),
                    .collective_payload_slot_bytes = collective_payload_slot_bytes,
                    .collective_payload_slot_capacity = collective_payload_slot_capacity,
                    .transfer_state = transfer_state,
                    .producer_layer_idx = layer_idx};
                graph_rebalance_plan_inserted = true;

                const bool producer_runs_in_maintenance_graph =
                    env.moe_rebalance.device_rebalance_maintenance_graph;
                const bool producer_can_use_collective_sideband =
                    !producer_runs_in_maintenance_graph &&
                    graph_rebalance_transfer_mode.value() ==
                        DeviceMoERebalanceTransferMode::CollectiveSidebandPayload &&
                    graph_rebalance_can_have_allreduce_anchor &&
                    local_tp_ctx &&
                    local_tp_ctx->supportsCollectiveSidebandOnStreamGraphCapture();

                if (producer_can_use_collective_sideband)
                {
                    graph_rebalance_collect_node =
                        prefix + "moe_device_rebalance_collect_state";
                    MoEDeviceRebalanceStage::Params collect_params;
                    collect_params.device_id = device;
                    collect_params.tp_ctx = local_tp_ctx;
                    collect_params.moe_runtime_table = moe_runtime_table;
                    collect_params.tp_device_idx = config_.tp_device_idx;
                    collect_params.config = rebalance_config;
                    collect_params.local_transfer_slots = local_transfer_slots;
                    collect_params.local_transfer_slot_count = local_transfer_slot_count;
                    collect_params.collective_payload_slot_bytes = collective_payload_slot_bytes;
                    collect_params.collective_payload_slot_capacity = collective_payload_slot_capacity;
                    collect_params.stage_name = graph_rebalance_collect_node;
                    collect_params.workspace_name = rebalance_workspace;
                    collect_params.phase = DeviceMoERebalanceStagePhase::CollectState;
                    collect_params.transfer_mode = graph_rebalance_transfer_mode.value();
                    collect_params.transfer_state = transfer_state;

                    graph.addNode(graph_rebalance_collect_node,
                                  ComputeStageFactory::createMoEDeviceRebalance(collect_params),
                                  device);
                    if (!graph_rebalance_producer_node_name.empty())
                    {
                        graph.addDependency(graph_rebalance_collect_node,
                                            graph_rebalance_producer_node_name);
                    }
                }

                if (!producer_runs_in_maintenance_graph)
                {
                    const std::string plan_node = prefix + "moe_device_rebalance_plan_copy";

                    MoEDeviceRebalanceStage::Params plan_params;
                    plan_params.device_id = device;
                    plan_params.tp_ctx = local_tp_ctx;
                    plan_params.moe_runtime_table = moe_runtime_table;
                    plan_params.tp_device_idx = config_.tp_device_idx;
                    plan_params.config = rebalance_config;
                    plan_params.local_transfer_slots = local_transfer_slots;
                    plan_params.local_transfer_slot_count = local_transfer_slot_count;
                    plan_params.collective_payload_slot_bytes = collective_payload_slot_bytes;
                    plan_params.collective_payload_slot_capacity = collective_payload_slot_capacity;
                    plan_params.stage_name = plan_node;
                    plan_params.workspace_name = rebalance_workspace;
                    plan_params.phase = DeviceMoERebalanceStagePhase::PlanAndCopy;
                    plan_params.transfer_mode = graph_rebalance_transfer_mode.value();
                    plan_params.join_transfer_stream_after_copy =
                        producer_runs_in_maintenance_graph;
                    plan_params.transfer_state = transfer_state;

                    if (producer_can_use_collective_sideband)
                    {
                        plan_params.phase =
                            DeviceMoERebalanceStagePhase::PlanAndCopyAfterSideband;
                        graph_rebalance_plan_after_sideband_node = plan_node;
                        graph_rebalance_plan_after_sideband_params = plan_params;
                    }
                    else
                    {
                        graph.addNode(plan_node,
                                      ComputeStageFactory::createMoEDeviceRebalance(plan_params),
                                      device);
                        graph.addDependency(plan_node, prefix + "ffn_norm");
                    }
                }

                LOG_DEBUG("[Qwen35MoEGraph] Added graph-side MoE device rebalance producer"
                          << " layer=" << layer_idx
                          << " device=" << device.to_string()
                          << " participant=" << config_.tp_device_idx
                          << " degree=" << local_tp_ctx->degree()
                          << " hot_replica_cap=" << hot_replica_cap
                          << " transfer_mode="
                          << graphRebalanceTransferModeName(graph_rebalance_transfer_mode.value())
                          << " transfer_slots=" << local_transfer_slot_count
                          << " collective_payload_slot_capacity=" << collective_payload_slot_capacity
                          << " window_tokens=" << rebalance_config.window_size_tokens
                          << " layer_window_start=" << rebalance_config.layer_window_start
                          << " layer_window_count=" << rebalance_config.layer_window_count
                          << " layer_wave_count=" << rebalance_config.layer_wave_count
                          << " producer_mode="
                          << (producer_runs_in_maintenance_graph ? "maintenance_graph" : "inline_decode_graph")
                          << " context=" << (context ? context : ""));

                const bool resident_boundary_apply_candidate =
                    producer_runs_in_maintenance_graph &&
                    !deviceMoERebalanceModeUsesTransferSlots(graph_rebalance_transfer_mode.value());
                const bool producer_covers_current_layer =
                    rebalance_config.layer_window_count == 1u &&
                    rebalance_config.layer_window_start == static_cast<uint32_t>(std::max(0, layer_idx));
                if (!producer_covers_current_layer && !resident_boundary_apply_candidate)
                    return {};
            }

            const auto binding_it = moe_graph_rebalance_bindings_.find(domain_key);
            if (binding_it == moe_graph_rebalance_bindings_.end())
                return {};
            /*
             * Maintenance-graph mode applies ready waves from the route kernel
             * itself so steady-state decode does not pay an extra per-token
             * apply-stage launch.
             */
            if (env.moe_rebalance.device_rebalance_maintenance_graph)
            {
                return {};
            }
            const bool decode_apply_poll_enabled =
                env.moe_rebalance.device_rebalance_decode_apply_poll;
            if (!decode_apply_poll_enabled)
            {
                return {};
            }

            const bool needs_transfer_slot_apply =
                deviceMoERebalanceModeUsesTransferSlots(binding_it->second.transfer_mode) &&
                binding_it->second.local_transfer_slot_count > 0;
            const bool needs_deferred_resident_apply =
                hasDeviceMoERebalanceFlag(
                    binding_it->second.config.flags,
                    DeviceMoERebalanceFlags::DeferRuntimeApply);
            if (!needs_transfer_slot_apply && !needs_deferred_resident_apply)
            {
                return {};
            }
            const bool boundary_apply =
                needs_deferred_resident_apply || needs_transfer_slot_apply;
            if (boundary_apply && !last_device_rebalance_decode_layer)
            {
                return {};
            }

            DeviceMoEExpertDirectoryEntry *apply_transfer_slots = nullptr;
            uint32_t apply_transfer_slot_count = 0;
            std::shared_ptr<DeviceMoERebalanceTransferState> apply_transfer_state =
                binding_it->second.transfer_state;
            if (needs_transfer_slot_apply)
            {
                const auto transfer_directory_it =
                    moe_transfer_slot_directories_.find(binding_it->second.transfer_key);
                const auto transfer_state_it =
                    moe_rebalance_transfer_states_.find(
                        binding_it->second.transfer_key + ":workspace=" +
                        binding_it->second.workspace_name);
                if (transfer_directory_it == moe_transfer_slot_directories_.end() ||
                    !transfer_directory_it->second ||
                    transfer_state_it == moe_rebalance_transfer_states_.end() ||
                    !transfer_state_it->second)
                {
                    throw std::runtime_error(
                        "Qwen35 MoE graph-side rebalance lost its device transfer binding for layer " +
                        std::to_string(layer_idx) + " on " + device.to_string());
                }
                apply_transfer_slots = transfer_directory_it->second->deviceEntries();
                apply_transfer_slot_count = transfer_directory_it->second->slotCount();
                apply_transfer_state = transfer_state_it->second;
            }

            graph_rebalance_apply_node = prefix + "moe_device_rebalance_apply";
            MoEDeviceRebalanceStage::Params apply_params;
            apply_params.device_id = device;
            apply_params.tp_ctx = local_tp_ctx;
            apply_params.moe_runtime_table = moe_runtime_table;
            apply_params.tp_device_idx = config_.tp_device_idx;
            apply_params.config = rebalance_config;
            apply_params.local_transfer_slots = apply_transfer_slots;
            apply_params.local_transfer_slot_count = apply_transfer_slot_count;
            apply_params.collective_payload_slot_bytes =
                binding_it->second.collective_payload_slot_bytes;
            apply_params.collective_payload_slot_capacity =
                binding_it->second.collective_payload_slot_capacity;
            apply_params.stage_name = graph_rebalance_apply_node;
            apply_params.workspace_name = binding_it->second.workspace_name;
            apply_params.phase = DeviceMoERebalanceStagePhase::Apply;
            apply_params.transfer_mode = binding_it->second.transfer_mode;
            apply_params.apply_layer_idx = boundary_apply ? -1 : layer_idx;
            apply_params.transfer_state = apply_transfer_state;

            graph.addNode(graph_rebalance_apply_node,
                          ComputeStageFactory::createMoEDeviceRebalance(apply_params),
                          device);
            graph.addDependency(graph_rebalance_apply_node, prefix + "ffn_norm");
            graph.addDependency(prefix + "moe_routing", graph_rebalance_apply_node);

            LOG_DEBUG("[Qwen35MoEGraph] Added graph-side MoE device rebalance layer apply"
                      << " producer_layer=" << binding_it->second.producer_layer_idx
                      << " layer=" << layer_idx
                      << " apply_layer=" << apply_params.apply_layer_idx
                      << " device=" << device.to_string()
                      << " participant=" << config_.tp_device_idx
                      << " context=" << (context ? context : ""));

            return graph_rebalance_apply_node;
        };

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
        /*
         * Routing and routed-expert execution form one graph-local producer /
         * consumer transaction. They share Q8 hidden rows and device route
         * metadata through this explicit owner, while every separately built
         * main graph, MTP sidecar, and layer gets a different owner.
         */
        auto routed_pipeline_kernel_owner =
            std::make_shared<MoERoutedPipelineKernelOwner>();
        TensorBase *routing_indices = buffers.get(buffers.idFor(BufferId::MOE_EXPERT_INDICES));
        TensorBase *routing_weights = buffers.get(buffers.idFor(BufferId::MOE_EXPERT_WEIGHTS));
        int expert_intermediate = config_.moe.intermediate_size;
        if (expert_intermediate == 0 && layer.moe_gate_exps)
        {
            // gate_exps shape: [num_experts, intermediate, d_model] or rows=num_experts*intermediate
            size_t total_rows = layer.moe_gate_exps->rows();
            expert_intermediate = static_cast<int>(total_rows / config_.moe.num_experts);
        }
        if (expert_intermediate <= 0)
        {
            throw std::runtime_error(
                "Qwen35 MoE graph requires an explicit positive routed-expert "
                "intermediate size when raw expert parents are registry-owned");
        }

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
            route_params.decode_histogram =
                mtp_sidecar_context ||
                        device_side_graph_rebalance_candidate ||
                        !collect_runtime_histogram
                    ? nullptr
                    : config_.moe.decode_histogram;
            route_params.host_logical_row_count =
                device.is_cpu() && !mtp_sidecar_context
                    ? total_tokens
                    : 0;
            route_params.moe_runtime_table = moe_runtime_table;
            route_params.collect_device_runtime_histogram =
                collect_runtime_histogram;
            route_params.decode_route_publication =
                use_expert_overlay && device.is_gpu() && !moe_runtime_table
                    ? MoEDecodeRoutePublicationPolicy::
                          FixedCapacityOverlayTicket
                    : MoEDecodeRoutePublicationPolicy::DeviceRuntimeTable;
            route_params.routed_row_execution_policy =
                routed_row_execution_policy;
            route_params.force_grouped_verifier_prefill_for_decode =
                forceGroupedMoEVerifierPrefill(device);
            route_params.absolute_position_ids_device =
                device.is_gpu()
                    ? absolute_position_ids_device
                    : nullptr;
            route_params.routed_pipeline_kernel_owner =
                routed_pipeline_kernel_owner;
            route_params.force_decode_equivalent_verifier_prefill =
                forceDecodeEquivalentMoERouting(device);
            route_params.grouped_verifier_histogram_role =
                selectMoEGroupedVerifierHistogramRole(
                    forceGpuSmallMMainVerifierPrefill(device) &&
                    use_expert_overlay &&
                    !masked_local_tp_overlay_decode_runtime_table &&
                    !captured_distributed_overlay_runtime_table &&
                    !full_local_tp_replicated_overlay_decode_runtime_table,
                    collect_runtime_histogram);
            route_params.active_row_count_device =
                device.is_gpu() && batch_size == 1
                    ? sequence_lengths_device
                    : nullptr;
            route_params.output_indices = routing_indices;
            route_params.output_weights = routing_weights;
            route_params.input_buffer_id = buffers.idFor(BufferId::NORMALIZED);
            route_params.output_indices_buffer_id = buffers.idFor(BufferId::MOE_EXPERT_INDICES);
            route_params.output_weights_buffer_id = buffers.idFor(BufferId::MOE_EXPERT_WEIGHTS);

            if (device_side_graph_rebalance_candidate &&
                env.moe_rebalance.device_rebalance_maintenance_graph &&
                device_rebalance_decode_layer &&
                (first_device_rebalance_decode_layer ||
                 last_device_rebalance_decode_layer))
            {
                if (first_device_rebalance_decode_layer &&
                    shouldCollectGraphRebalanceTransferSpecs() &&
                    !graph_rebalance_transfer_profile.has_value())
                {
                    graph_rebalance_transfer_profile =
                        collectGraphRebalanceTransferProfile();
                }

                if (first_device_rebalance_decode_layer)
                    (void)maybeInsertGraphSideRebalance(
                        "MoE routing ready-wave apply piggyback",
                        std::string{});

                const auto binding_it =
                    moe_graph_rebalance_bindings_.find(graphRebalanceDomainKey());
                if (binding_it != moe_graph_rebalance_bindings_.end())
                {
                    const auto &binding = binding_it->second;
                    const bool needs_transfer_slot_apply =
                        deviceMoERebalanceModeUsesTransferSlots(binding.transfer_mode) &&
                        binding.local_transfer_slot_count > 0;
                    const bool needs_deferred_resident_apply =
                        hasDeviceMoERebalanceFlag(
                            binding.config.flags,
                            DeviceMoERebalanceFlags::DeferRuntimeApply);
                    if (needs_transfer_slot_apply || needs_deferred_resident_apply)
                    {
                        const auto *durable_runtime =
                            dynamic_cast<const DeviceMoERuntimeTable *>(
                                moe_runtime_table);
                        const bool durable_epoch_bound =
                            durable_runtime &&
                            durable_runtime->usesOverlayEpochTicket();
                        if (last_device_rebalance_decode_layer &&
                            !durable_epoch_bound)
                        {
                            route_params.device_rebalance_route_apply = true;
                            route_params.device_rebalance_workspace_name =
                                binding.workspace_name;
                            route_params.device_rebalance_config = binding.config;
                            route_params.device_rebalance_local_transfer_slots =
                                binding.local_transfer_slots;
                            route_params.device_rebalance_local_transfer_slot_count =
                                binding.local_transfer_slot_count;
                            route_params.device_rebalance_plan_capacity =
                                static_cast<uint32_t>(
                                    transferPlanCapacityForBinding(binding));
                            route_params.device_rebalance_command_buffer_count =
                                static_cast<uint32_t>(
                                    commandBufferCountForBinding(binding));
                            route_params.device_rebalance_apply_layer_idx = -1;
                        }
                    }
                }
            }

            graph.addNode(prefix + "moe_routing",
                          ComputeStageFactory::createMoERouting(route_params),
                          device);
            graph.addDependency(prefix + "moe_routing", prefix + "ffn_norm");
        }

        // =====================================================================
        // Stage 3: MoE Expert Compute (routed expert SwiGLU FFN)
        // =====================================================================
        TensorBase *moe_output = buffers.get(buffers.idFor(BufferId::MOE_COMBINED_OUTPUT));
        TensorBase *canonical_route_contributions =
            buffers.get(
                buffers.idFor(
                    BufferId::MOE_CANONICAL_ROUTE_CONTRIBUTIONS));
        TensorBase *shared_output = buffers.get(buffers.idFor(BufferId::MOE_SHARED_EXPERT_OUTPUT));
        bool moe_combined_output_ready = false;
        std::string shared_ffn_last; // Track last shared expert stage (empty if no shared expert)
        /* Retain the root-only publisher identity so trace-only checkpoints can
         * localize the exact captured sibling immediately before it. */
        /*
         * A distributed sparse overlay and dense NodeTP share one rank-wide
         * collective order.  The routed branch is constructed before the
         * shared branch, but the two are otherwise graph siblings.  Retain a
         * typed construction fact so the later shared-expert collective can
         * depend on completion of the sparse protocol on every rank; relying
         * on topological insertion order deadlocks when root and peer graphs
         * contain different payload-authority nodes.
         */
        bool routed_overlay_has_distributed_sparse_protocol = false;

        auto plannedSharedExpertDevice = [&]() -> DeviceId
        {
            DeviceId shared_device = device;
            if (overlay_runtime_plan)
            {
                const auto &shared_domain = overlay_runtime_plan->sharedExpertDomain();
                const bool distributed_participant_graph =
                    config_.moe.overlay_mpi_ctx &&
                    config_.moe.overlay_mpi_ctx->world_size() > 1;
                /*
                 * Remote expert ranks execute a participant-local graph whose
                 * only authoritative outputs are the sparse routed rows.  The
                 * surrounding full graph is temporary construction scaffolding
                 * until the dedicated participant graph owns this branch; it
                 * must never reach across runtimes to execute the root's shared
                 * expert.  Keep that non-authoritative branch on the exact local
                 * device so every operation still has a valid device context.
                 */
                if (distributed_participant_graph &&
                    !shared_domain.primary_owned_by_current_rank)
                {
                    shared_device = device;
                }
                else
                {
                    shared_device =
                        overlay_runtime_plan->sharedExpertDeviceForMVP(layer_idx);
                }
                if (domainContainsDevice(shared_domain, device))
                    shared_device = device;
            }
            return shared_device;
        };
        const bool has_shared_expert_branch =
            layer.shared_expert_gate && layer.shared_expert_up &&
            layer.shared_expert_down && shared_output;
        const DeviceId planned_shared_device =
            has_shared_expert_branch ? plannedSharedExpertDevice() : device;

        const RoutedExpertTier *local_tp_apportioned_tier = nullptr;
        const bool local_tp_apportioned_fast_candidate =
            use_expert_overlay &&
            canUseLocalTPExpertIdApportionedFastPath(
                *overlay_plan,
                device,
                &local_tp_apportioned_tier);
        const RoutedExpertTier *local_tp_replicated_tier =
            resolved_local_tp_replicated_tier;
        const bool local_tp_replicated_fast_candidate =
            resolved_local_tp_replicated_fast_candidate;

        auto needsMoEParticipantAllreduce = [&]() -> bool
        {
            return config_.tp_ctx && config_.tp_ctx->degree() > 1 &&
                   routed_row_execution_policy ==
                       RoutedExpertRowExecutionPolicy::ParticipantAssigned;
        };
        const bool shared_expert_requires_tp_allreduce =
            has_shared_expert_branch && needsTPAllreduce() &&
            denseTPAllreduceEnabledForCurrentGraph();

        /*
         * The grouped ROCm verifier has enough independent route work to make
         * one serial top-k loop per output lane uneconomical. A single-device
         * graph therefore publishes route dots independently and declares the
         * exact ordered reducer as its next node. This is a graph policy, not a
         * runtime branch: capture records one fixed producer/reducer topology,
         * all storage is persistent, and the reducer preserves serial decode's
         * increasing-route FP32 addition order byte-for-byte.
         */
        const int graph_participant_count =
            config_.tp_ctx ? config_.tp_ctx->degree() : 1;
        const MoERouteAccumulationPolicy route_accumulation_policy =
            selectMoERouteAccumulationPolicy(
                MoERouteAccumulationSelection{
                    .backend = device.type,
                    .participant_count = graph_participant_count,
                    .workload = forceGroupedMoEVerifierPrefill(device)
                                    ? MoERouteAccumulationWorkload::
                                          GroupedVerifier
                                    : MoERouteAccumulationWorkload::Ordinary});

        /*
         * LocalTP expert ownership must never shape the FP32 route addition
         * tree. Apportioned paths therefore publish every original route into
         * an independent slot. CPU graphs allreduce raw expert rows and repeat
         * the serial weighted-FMA fold on every participant. GPU graphs use a
         * rooted reduction of preweighted slots followed by one ordered root
         * fold and compact broadcast. When the shared GPU branch is
         * input-parallel, rank-addressed shared banks join that same rooted
         * transaction. The complete transport/arithmetic choice is a typed
         * policy below rather than an implicit backend convention.
         */
        bool canonical_local_tp_route_publication = false;
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

            [[nodiscard]] bool usesPackedRouteGather() const noexcept
            {
                return policy == MoEParticipantPublicationPolicy::
                                     CanonicalRootedPackedRouteRows;
            }
        } canonical_publication_lowering;

        /**
         * Rooted continuation publication deferred until the shared branch.
         *
         * Presence means the routed branch deliberately did not broadcast its
         * intermediate row. The shared branch must reduce its partial row to
         * the same fixed authority, fuse gate+combine there, and publish only
         * the final tensor. Keeping this as one complete contract prevents a
         * later graph edit from accidentally consuming root-only bytes on a
         * non-root participant.
         */
        struct DeferredOverlayCombinedPublication
        {
            int root_device_index = -1;
            int participant_count = 0;
            std::string routed_terminal;
            std::string capture_wave_identity;

            /** @return Whether every fixed topology field is complete. */
            [[nodiscard]] bool valid() const noexcept
            {
                return root_device_index >= 0 && participant_count > 1 &&
                       root_device_index < participant_count &&
                       !routed_terminal.empty() &&
                       !capture_wave_identity.empty();
            }
        };
        std::optional<DeferredOverlayCombinedPublication>
            deferred_overlay_combined_publication;
        std::string captured_overlay_routed_unit_terminal;

        {
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
                expert_params.routed_pipeline_kernel_owner =
                    routed_pipeline_kernel_owner;
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
                expert_params.service_phase = routed_service_phase;
                expert_params.weight_descriptor_source =
                    dynamic_distributed_overlay_uses_mutable_descriptors ||
                            graphRebalanceDecodeUsesMutableDescriptors() ||
                            activeRuntimeBankUsesTransientLocalPayload(layer_idx)
                        ? MoEDecodeDescriptorSource::RuntimePlacementTable
                        : MoEDecodeDescriptorSource::StaticDescriptorTable;
                expert_params.runtime_decode_has_explicit_owner_metadata =
                    masked_local_tp_overlay_decode_runtime_table ||
                    captured_distributed_overlay_runtime_table ||
                    full_local_tp_replicated_overlay_decode_runtime_table ||
                    masked_local_tp_apportioned_decode_runtime_table;
                expert_params.force_grouped_verifier_prefill_for_decode =
                    forceGroupedMoEVerifierPrefill(stage_device);
                expert_params.grouped_verifier_histogram_role =
                    selectMoEGroupedVerifierHistogramRole(
                        forceGpuSmallMMainVerifierPrefill(stage_device),
                        collect_runtime_histogram);
                expert_params.force_decode_equivalent_verifier_prefill =
                    forceDecodeEquivalentMoEVerifier(stage_device);
                expert_params.absolute_position_ids_device =
                    stage_device.is_gpu()
                        ? absolute_position_ids_device
                        : nullptr;
                expert_params.active_row_count_device =
                    stage_device.is_gpu() && batch_size == 1
                        ? sequence_lengths_device
                        : nullptr;
                expert_params.my_socket_id =
                    std::max(0, config_.tp_device_idx);
                expert_params.participant_count =
                    config_.tp_ctx && config_.tp_ctx->degree() > 0
                        ? config_.tp_ctx->degree()
                        : std::max(1, expert_params.my_socket_id + 1);
                expert_params.routed_assignment_policy =
                    prefill_routed_expert_assignment_policy;
                expert_params.routed_row_execution_policy =
                    routed_row_execution_policy;
                if (local_tp_ctx &&
                    total_tokens > 1 &&
                    routed_row_execution_policy ==
                        RoutedExpertRowExecutionPolicy::ParticipantAssigned &&
                    prefill_routed_expert_assignment_policy == RoutedExpertAssignmentPolicy::LeastLoadedResident)
                {
                    expert_params.prefill_llep_tp_ctx = local_tp_ctx;
                    expert_params.prefill_llep_assignment_mode =
                        current_batch_llep_transfer_candidate
                            ? PrefillLLEPAssignmentMode::
                                  GraphPhasedCurrentBatch
                            : PrefillLLEPAssignmentMode::
                                  LogicalPositionResidentOnly;
                }

                if (config_.moe.routed_compute_policy == RoutedExpertComputePolicy::Apportioned)
                {
                    expert_params.local_expert_start = config_.moe.local_expert_start;
                    expert_params.local_expert_count = config_.moe.local_expert_count;
                    if (config_.moe.owner_participant_count > 1 &&
                        expert_params.expert_mask.empty())
                    {
                        /*
                         * Static ownership is derived by one shared authority
                         * for ordinary TP and graph-native overlays. Ordinal
                         * ownership reproduces the former contiguous range;
                         * random ownership uses a deterministic per-layer
                         * permutation and therefore cannot be represented by
                         * start/count alone.
                         */
                        expert_params.expert_mask =
                            routed_expert_ownership::expertMaskForParticipant(
                                config_.moe.num_experts,
                                config_.moe.owner_participant_count,
                                config_.moe.owner_participant_index,
                                layer_idx,
                                config_.moe.owner_order);
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

                // Set expert_registry for dynamic rebalancing registry updates
                if (model_ctx_)
                {
                    auto weight_mgr = model_ctx_->concreteWeightManager();
                    if (weight_mgr)
                        expert_params.expert_registry = &weight_mgr->expertGemmRegistry();
                }

                /*
                 * A graph-native overlay owns participant-scoped prepared
                 * engines. Its generic model-weight accessor is intentionally
                 * not an expert-slice authority: on a rank that also owns a CPU
                 * endpoint, the same canonical tensor name may denote that
                 * endpoint's much smaller packed slice. Resolve the registry
                 * first and make raw fallback structurally impossible.
                 */
                const bool registry_only_overlay =
                    stage_device.is_gpu() && !registry_domain_name.empty();

                // Standalone paths still derive their prepared engines from
                // the exact raw parent bound to this graph device.
                if (!registry_only_overlay &&
                    !MoEExpertComputeStage::extractExpertViews(expert_params))
                {
                    LOG_ERROR("[Qwen35MoEGraph] Failed to extract expert views for layer " << layer_idx);
                    return false;
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

                        LOG_TRACE("[Qwen35MoEGraph] Layer " << layer_idx
                                                            << ": populated expert GEMM engines from registry"
                                                            << (domain_scoped ? " domain=" + registry_domain_name : std::string())
                                                            << " complete_layer=" << complete_layer
                                                            << " active_masked_expert=" << has_active_masked_expert);
                    }

                    if (registry_only_overlay)
                    {
                        expert_params.gate_exps = nullptr;
                        expert_params.up_exps = nullptr;
                        expert_params.down_exps = nullptr;
                        expert_params.expert_gate_views.clear();
                        expert_params.expert_up_views.clear();
                        expert_params.expert_down_views.clear();
                        expert_params.expert_weight_resolution_policy =
                            MoEExpertWeightResolutionPolicy::
                                PreparedRegistryOnly;
                    }
                }
                else
                {
                    if (!MoEExpertComputeStage::prepareExpertGemmEngines(expert_params))
                        return false;
                }

                return true;
            };

            /**
             * Publish one graph-local participant's immutable initial RCU bank.
             *
             * Both the homogeneous fast path and a heterogeneous topology's
             * captured continuation-local branch consume prepared engines
             * directly through @ref MoEExpertComputeStage.  Neither constructs
             * a host-staged @ref MoELocalExpertStage, so graph construction must
             * install the exact registry-owned lifetimes here.  The resident
             * mask always comes from the canonical owner map; a broader
             * execution mask (for example a replicated decode policy) must
             * never become a second residency authority.
             */
            auto registerInitialOverlayParticipantBank =
                [&](const MoEExpertOwnerMap &owner_map,
                    int local_participant,
                    const MoEExpertComputeStage::Params &expert_params,
                    const std::string &path_name)
            {
                if (config_.moe.expert_overlay_residency_authority &&
                    !config_.moe.expert_overlay_participant_residency)
                {
                    throw std::runtime_error(
                        path_name +
                        " has a global residency authority but no process-local epoch-indexed prepared banks");
                }
                if (!config_.moe.expert_overlay_participant_residency)
                    return;
                if (!expert_params.expert_registry)
                {
                    throw std::runtime_error(
                        path_name +
                        " has no model-owned prepared engine registry");
                }

                const auto *participant =
                    owner_map.participantForId(local_participant);
                auto endpoint =
                    config_.moe.expert_overlay_participant_residency
                        ->endpoint(local_participant);
                if (!participant || !endpoint)
                {
                    throw std::runtime_error(
                        path_name + " owns participant p" +
                        std::to_string(local_participant) +
                        " but its process-local residency endpoint is missing");
                }

                const auto canonical_resident_mask =
                    owner_map.expertMaskForParticipant(
                        layer_idx,
                        local_participant,
                        config_.moe.num_experts);
                std::vector<MoEOverlayPreparedExpertTriplet>
                    prepared_triplets;
                std::string residency_error;
                if (!resolveMoEOverlayPreparedExpertTriplets(
                        *expert_params.expert_registry,
                        *participant,
                        layer_idx,
                        config_.moe.num_experts,
                        canonical_resident_mask,
                        prepared_triplets,
                        &residency_error) ||
                    !config_.moe.expert_overlay_participant_residency
                         ->registerInitialLayer(
                             local_participant,
                             layer_idx,
                             canonical_resident_mask,
                             prepared_triplets,
                             &residency_error))
                {
                    throw std::runtime_error(
                        path_name +
                        " could not install initial prepared residency bank for layer " +
                        std::to_string(layer_idx) + " participant " +
                        std::to_string(local_participant) + ": " +
                        residency_error);
                }
            };

            /*
             * Phase 9.8 production guard: the only accepted GPU verifier route
             * is split branch-local math.  Routed experts use the grouped MoE
             * verifier pipeline, while the shared expert uses its standalone
             * grouped table-prefill verifier path plus the normal gate/combine
             * stage.  The combined routed+shared owner is intentionally kept out
             * of the graph until it proves byte-identical to row-by-row serial
             * decode across the full model, not merely close on relaxed metrics.
            */
            const bool can_combine_shared_verifier = false;

            const RoutedExpertTier *local_tp_fast_tier =
                local_tp_apportioned_fast_candidate
                    ? local_tp_apportioned_tier
                    : local_tp_replicated_tier;
            const bool use_local_tp_routed_fast_path =
                local_tp_apportioned_fast_candidate ||
                phase_split_local_tp_apportioned_gpu_prefill ||
                local_tp_replicated_fast_candidate;

            if (use_local_tp_routed_fast_path)
            {
                auto owner_map_lifetime = std::make_shared<MoEExpertOwnerMap>(
                    MoEExpertOwnerMap::build(*overlay_plan));
                const int local_participant = participantIdForTierDevice(
                    *owner_map_lifetime,
                    0,
                    device);
                if (local_participant < 0)
                {
                    throw std::runtime_error(
                        "Qwen35 MoE LocalTP routed-expert fast path could not find graph-local participant for " +
                        device.to_string() + " in layer " + std::to_string(layer_idx));
                }

                const bool least_loaded_phase_split_prefill =
                    phase_split_local_tp_apportioned_gpu_prefill &&
                    prefill_routed_expert_assignment_policy ==
                        RoutedExpertAssignmentPolicy::LeastLoadedResident;
                auto participant_mask =
                    local_tp_replicated_fast_candidate &&
                            (!phase_split_local_tp_apportioned_gpu_prefill ||
                             least_loaded_phase_split_prefill)
                        ? std::vector<bool>(
                              static_cast<size_t>(config_.moe.num_experts),
                              true)
                        : owner_map_lifetime->expertMaskForParticipant(
                              layer_idx,
                              local_participant,
                              config_.moe.num_experts);
                if (!hasActiveExpertMask(participant_mask))
                {
                    throw std::runtime_error(
                        "Qwen35 MoE LocalTP routed-expert fast path produced an empty expert mask for participant " +
                        std::to_string(local_participant) + " in layer " + std::to_string(layer_idx));
                }

                auto expert_params = makeExpertParams(
                    moe_output,
                    buffers.idFor(BufferId::MOE_COMBINED_OUTPUT),
                    std::move(participant_mask),
                    device);
                expert_params.my_socket_id = local_participant;
                expert_params.participant_count =
                    participantCountForGraphNativeOverlay(
                        *owner_map_lifetime,
                        continuationRootParticipant(*overlay_plan));
                if (local_tp_ctx &&
                    total_tokens > 1 &&
                    routed_row_execution_policy ==
                        RoutedExpertRowExecutionPolicy::ParticipantAssigned &&
                    prefill_routed_expert_assignment_policy == RoutedExpertAssignmentPolicy::LeastLoadedResident)
                {
                    expert_params.prefill_llep_tp_ctx = local_tp_ctx;
                    expert_params.prefill_llep_assignment_mode =
                        current_batch_llep_transfer_candidate
                            ? PrefillLLEPAssignmentMode::
                                  GraphPhasedCurrentBatch
                            : PrefillLLEPAssignmentMode::
                                  LogicalPositionResidentOnly;
                }
                const std::string domain_name = local_tp_fast_tier ? local_tp_fast_tier->domain : std::string{};
                if (!prepareExpertParams(
                        expert_params,
                        device,
                        "LocalTP routed-expert fast path participant " +
                            std::to_string(local_participant),
                        domain_name))
                {
                    throw std::runtime_error(
                        "Qwen35 MoE graph failed to prepare LocalTP routed-expert fast-path parameters for layer " +
                        std::to_string(layer_idx) + " on " + device.to_string());
                }

                registerInitialOverlayParticipantBank(
                    *owner_map_lifetime,
                    local_participant,
                    expert_params,
                    "Qwen35 MoE LocalTP fast path");
                const bool least_loaded_prefill_runtime_grouping =
                    moe_runtime_table &&
                    total_tokens > 1 &&
                    prefill_routed_expert_assignment_policy == RoutedExpertAssignmentPolicy::LeastLoadedResident;
                if (least_loaded_prefill_runtime_grouping)
                {
                    const int participant_count = expert_params.participant_count;
                    const auto owner_participants =
                        ownerParticipantsFromMap(
                            *owner_map_lifetime,
                            layer_idx,
                            config_.moe.num_experts);
                    if (!initializeMaskedLocalDecodeRuntimeTable(
                            moe_runtime_table,
                            layer_idx,
                            config_.moe.num_experts,
                            config_.moe.top_k,
                            config_.d_model,
                            expert_intermediate,
                            expert_params.expert_mask,
                            local_participant,
                            participant_count,
                            owner_participants,
                            owner_participants,
                            expert_params.prepared_gate_gemm,
                            expert_params.prepared_up_gemm,
                            expert_params.prepared_down_gemm,
                            device_state_publication_stream,
                            prefill_routed_expert_assignment_policy == RoutedExpertAssignmentPolicy::LeastLoadedResident,
                            "LocalTP expert-ID-apportioned LLEP grouped prefill runtime bank"))
                    {
                        throw std::runtime_error(
                            "Qwen35 MoE graph failed to initialize masked LocalTP LLEP prefill runtime table for layer " +
                            std::to_string(layer_idx) + " on " + device.to_string());
                    }
                    expert_params.use_runtime_row_grouping = true;
                }

                if (prefill_llep_transfer_candidate &&
                    (expert_params.use_runtime_row_grouping ||
                     prefix_runtime_device_rehydration))
                {
                    attachPrefillLLEPTransferBinding(
                        expert_params,
                        "LocalTP expert-ID-apportioned LLEP grouped prefill");
                }

                if (current_batch_llep_transfer_candidate)
                {
                    if (!expert_params.use_runtime_row_grouping ||
                        expert_params.prefill_llep_assignment_mode !=
                            PrefillLLEPAssignmentMode::
                                GraphPhasedCurrentBatch ||
                        !expert_params.prefill_llep_tp_ctx ||
                        !expert_params.prefill_llep_transfer_slots ||
                        expert_params.prefill_llep_transfer_slot_count == 0 ||
                        expert_params.prefill_llep_payload_slot_bytes == 0 ||
                        expert_params.prefill_llep_payload_slot_capacity == 0)
                    {
                        throw std::logic_error(
                            "Qwen35 MoE graph-phased current-batch LLEP has an incomplete transfer binding for layer " +
                            std::to_string(layer_idx) + " on " +
                            device.to_string());
                    }

                    auto make_current_batch_phase =
                        [&](GPUCurrentBatchLLEPPhase phase,
                            const std::string &stage_name)
                    {
                        MoEGPUCurrentBatchLLEPStage::Params params;
                        params.device_id = device;
                        params.phase = phase;
                        params.tp_ctx = expert_params.prefill_llep_tp_ctx;
                        params.moe_runtime_table =
                            expert_params.moe_runtime_table;
                        params.routing_indices =
                            expert_params.routing_indices;
                        params.routing_weights =
                            expert_params.routing_weights;
                        params.routing_indices_buffer_id =
                            expert_params.routing_indices_buffer_id;
                        params.routing_weights_buffer_id =
                            expert_params.routing_weights_buffer_id;
                        params.tp_device_idx = config_.tp_device_idx;
                        params.layer_idx = layer_idx;
                        params.seq_len = total_tokens;
                        params.num_experts = config_.moe.num_experts;
                        params.top_k = config_.moe.top_k;
                        params.config =
                            expert_params.prefill_llep_rebalance_config;
                        params.local_transfer_slots =
                            expert_params.prefill_llep_transfer_slots;
                        params.local_transfer_slot_count =
                            expert_params.prefill_llep_transfer_slot_count;
                        params.payload_slot_bytes =
                            expert_params.prefill_llep_payload_slot_bytes;
                        params.payload_slot_capacity =
                            expert_params.prefill_llep_payload_slot_capacity;
                        params.transfer_mode =
                            expert_params.prefill_llep_transfer_mode;
                        params.workspace_name =
                            expert_params.prefill_llep_workspace_name;
                        params.stage_name = stage_name;
                        return params;
                    };

                    current_batch_llep_plan_node =
                        prefix + "moe_current_batch_llep_plan_pack";
                    current_batch_llep_apply_node =
                        prefix +
                        "moe_current_batch_llep_unpack_apply_assign";
                    current_batch_llep_expert_node =
                        prefix + "moe_expert_ffn_overlay_fast";
                    current_batch_llep_plan_params =
                        make_current_batch_phase(
                            GPUCurrentBatchLLEPPhase::PlanAndPack,
                            current_batch_llep_plan_node);
                    current_batch_llep_apply_params =
                        make_current_batch_phase(
                            GPUCurrentBatchLLEPPhase::
                                UnpackApplyAndAssign,
                            current_batch_llep_apply_node);
                }

                if (moe_runtime_table &&
                    masked_local_tp_overlay_decode_runtime_table &&
                    expert_params.replica_set.num_replicated == 0)
                {
                    const int participant_count = expert_params.participant_count;
                    const auto owner_participants =
                        ownerParticipantsFromMap(
                            *owner_map_lifetime,
                            layer_idx,
                            config_.moe.num_experts);
                    if (!initializeMaskedLocalDecodeRuntimeTable(
                            moe_runtime_table,
                            layer_idx,
                            config_.moe.num_experts,
                            config_.moe.top_k,
                            config_.d_model,
                            expert_intermediate,
                            expert_params.expert_mask,
                            local_participant,
                            participant_count,
                            owner_participants,
                            owner_participants,
                            expert_params.prepared_gate_gemm,
                            expert_params.prepared_up_gemm,
                            expert_params.prepared_down_gemm,
                            device_state_publication_stream,
                            /*allow_existing_dynamic_bank=*/true,
                            "LocalTP expert-ID-apportioned masked GPU decode graph build"))
                    {
                        throw std::runtime_error(
                            "Qwen35 MoE graph failed to initialize masked LocalTP decode runtime table for layer " +
                            std::to_string(layer_idx) + " on " + device.to_string());
                    }
                    if (total_tokens > 1 && forceGroupedMoEVerifierPrefill(device))
                    {
                        expert_params.use_runtime_row_grouping = true;
                    }
                }
                else if (moe_runtime_table &&
                         full_local_tp_replicated_overlay_decode_runtime_table)
                {
                    const int participant_count = expert_params.participant_count;
                    const auto owner_participants =
                        ownerParticipantsFromMap(
                            *owner_map_lifetime,
                            layer_idx,
                            config_.moe.num_experts);
                    if (!initializeFullLocalDecodeRuntimeTable(
                            moe_runtime_table,
                            layer_idx,
                            config_.moe.num_experts,
                            config_.moe.top_k,
                            config_.d_model,
                            expert_intermediate,
                            FullLocalDecodeRuntimePolicy::FullyReplicatedLocalTP,
                            local_participant,
                            participant_count,
                            owner_participants,
                            expert_params.prepared_gate_gemm,
                            expert_params.prepared_up_gemm,
                            expert_params.prepared_down_gemm,
                            device_state_publication_stream,
                            "LocalTP replicated GPU decode graph build"))
                    {
                        throw std::runtime_error(
                            "Qwen35 MoE graph failed to initialize replicated LocalTP decode runtime table for layer " +
                            std::to_string(layer_idx) + " on " +
                            device.to_string());
                    }
                    if (total_tokens > 1 &&
                        forceGroupedMoEVerifierPrefill(device))
                    {
                        expert_params.use_runtime_row_grouping = true;
                    }
                }

                /*
                 * Routed-expert ownership is an independent parallelism axis
                 * from dense tensor parallelism. A graph may deliberately
                 * replicate all dense branches while still apportioning MoE
                 * experts across CPU participants. Gate canonical route
                 * publication on the typed row-ownership contract itself;
                 * using needsTPAllreduce() here would silently restore the
                 * ownership-dependent compact-sum path whenever dense TP is
                 * disabled.
                 */
                const bool participant_routes_require_collective =
                    needsMoEParticipantAllreduce();
                if (!participant_routes_require_collective)
                {
                    canonical_publication_lowering.policy =
                        MoEParticipantPublicationPolicy::
                            IndependentBranchCollectives;
                }
                else if (device.is_cpu())
                {
                    canonical_publication_lowering.policy =
                        MoEParticipantPublicationPolicy::
                            CanonicalRootedPackedRouteRows;
                }
                else if (current_batch_llep_transfer_candidate)
                {
                    /* Current-batch foreign expert rows cannot execute until
                     * their payload arrives. Qwen's shared-expert allreduce is
                     * the only existing post-router collective that is early
                     * enough to carry that payload. Keep shared publication
                     * independent in this mode and retain the exact routed-slot
                     * rooted reduction after expert compute. */
                    if (!shared_expert_requires_tp_allreduce ||
                        !has_shared_expert_branch ||
                        !layer.shared_expert_gate_inp ||
                        planned_shared_device != device)
                    {
                        throw std::runtime_error(
                            "Qwen35 MoE graph-phased current-batch LLEP requires an input-parallel local shared-expert collective anchor for layer " +
                            std::to_string(layer_idx) + " on " +
                            device.to_string());
                    }
                    canonical_publication_lowering.policy =
                        MoEParticipantPublicationPolicy::
                            CanonicalRootedRouteSlots;
                }
                else if (shared_expert_requires_tp_allreduce &&
                         has_shared_expert_branch &&
                         layer.shared_expert_gate_inp &&
                         planned_shared_device == device)
                {
                    canonical_publication_lowering.policy =
                        MoEParticipantPublicationPolicy::
                            CanonicalRootedRankBanks;
                }
                else
                {
                    canonical_publication_lowering.policy =
                        MoEParticipantPublicationPolicy::
                            CanonicalRootedRouteSlots;
                }
                canonical_local_tp_route_publication =
                    canonical_publication_lowering.
                        usesCanonicalRouteSlots();
                if (routed_row_execution_policy ==
                        RoutedExpertRowExecutionPolicy::FullyReplicatedLocal &&
                    canonical_local_tp_route_publication)
                {
                    throw std::logic_error(
                        "Qwen35 MoE fully replicated local row execution cannot publish canonical route contributions");
                }
                if (canonical_local_tp_route_publication)
                {
                    if (!canonical_route_contributions)
                    {
                        throw std::runtime_error(
                            "Qwen35 MoE LocalTP canonical route buffer is missing for layer " +
                            std::to_string(layer_idx) + " on " +
                            device.to_string());
                    }
                    expert_params.canonical_route_contributions =
                        canonical_route_contributions;
                    expert_params.canonical_route_arithmetic =
                        device.is_cpu()
                            ? MoECanonicalRouteArithmeticPolicy::
                                  UnweightedExpertRowThenOrderedFMA
                            : MoECanonicalRouteArithmeticPolicy::
                                  PreweightedContributionThenOrderedAdd;
                    expert_params.canonical_route_layout =
                        canonical_publication_lowering.
                                usesPackedRouteGather()
                            ? MoECanonicalRoutePublicationLayout::
                                  PackedIndexedRouteRows
                            : MoECanonicalRoutePublicationLayout::
                                  DenseOriginalRouteSlots;
                    expert_params.canonical_route_contributions_buffer_id =
                        buffers.idFor(
                            BufferId::MOE_CANONICAL_ROUTE_CONTRIBUTIONS);
                }

                if (current_batch_llep_plan_params.has_value())
                {
                    if (current_batch_llep_plan_node.empty())
                    {
                        throw std::logic_error(
                            "Qwen35 MoE current-batch LLEP plan has no graph node identity");
                    }
                    graph.addNode(
                        current_batch_llep_plan_node,
                        ComputeStageFactory::createMoEGPUCurrentBatchLLEP(
                            *current_batch_llep_plan_params),
                        device);
                    graph.addDependency(
                        current_batch_llep_plan_node,
                        prefix + "moe_routing");
                }

                graph.addNode(prefix + "moe_expert_ffn_overlay_fast",
                              ComputeStageFactory::createMoEExpertCompute(expert_params),
                              device);
                const std::string rebalance_apply_dependency =
                    maybeInsertGraphSideRebalance(
                        "LocalTP expert-ID-apportioned fast path",
                        prefix + "moe_expert_ffn_overlay_fast");
                graph.addDependency(prefix + "moe_expert_ffn_overlay_fast",
                                    rebalance_apply_dependency.empty()
                                        ? prefix + "moe_routing"
                                        : rebalance_apply_dependency);
                ffn_terminal = prefix + "moe_expert_ffn_overlay_fast";

                if (needsMoEParticipantAllreduce())
                {
                    TensorBase *allreduce_buffer =
                        canonical_local_tp_route_publication
                            ? canonical_route_contributions
                            : moe_output;
                    const BufferId allreduce_buffer_id =
                        canonical_local_tp_route_publication
                            ? buffers.idFor(
                                  BufferId::MOE_CANONICAL_ROUTE_CONTRIBUTIONS)
                            : buffers.idFor(BufferId::MOE_COMBINED_OUTPUT);
                    const size_t allreduce_count =
                        static_cast<size_t>(total_tokens) *
                        static_cast<size_t>(config_.d_model) *
                        (canonical_local_tp_route_publication
                             ? static_cast<size_t>(config_.moe.top_k) +
                                   (canonical_publication_lowering.usesRankBanks()
                                        ? static_cast<size_t>(
                                              expert_params.participant_count)
                                        : size_t{0})
                             : size_t{1});
                    std::string ar_name;
                    if (!canonical_local_tp_route_publication)
                    {
                        ar_name = prefix + "moe_expert_overlay_fast_allreduce";
                    }
                    else if (canonical_publication_lowering.usesRankBanks())
                    {
                        ar_name =
                            prefix + "moe_canonical_publication_reduce_to_root";
                    }
                    else if (canonical_publication_lowering.
                                 usesPackedRouteGather())
                    {
                        ar_name =
                            prefix + "moe_canonical_routes_gather_to_root";
                    }
                    else if (canonical_publication_lowering.
                                 usesRootedCollective())
                    {
                        ar_name =
                            prefix + "moe_canonical_routes_reduce_to_root";
                    }
                    else
                    {
                        throw std::logic_error(
                            "Qwen35 MoE canonical publication policy has no "
                            "collective lowering for layer " +
                            std::to_string(layer_idx));
                    }
                    auto rebalance_sidebands =
                        takeGraphRebalanceSidebandsForAllreduce();
                    std::unique_ptr<IComputeStage> collective_stage;
                    int canonical_route_root_participant = -1;
                    if (canonical_publication_lowering.usesPackedRouteGather())
                    {
                        if (!rebalance_sidebands.empty() ||
                            !graph_rebalance_collect_node.empty() ||
                            graph_rebalance_plan_after_sideband_params.has_value() ||
                            graph_rebalance_pack_payload_params.has_value() ||
                            graph_rebalance_unpack_payload_params.has_value())
                        {
                            throw std::logic_error(
                                "Qwen35 MoE CPU packed route publication cannot "
                                "consume GPU graph-side rebalance sidebands for layer " +
                                std::to_string(layer_idx));
                        }
                        canonical_route_root_participant = 0;
                        MoECanonicalRouteGatherStage::Params gather_params;
                        gather_params.device_id = device;
                        gather_params.tp_ctx = config_.tp_ctx;
                        gather_params.packed_route_records =
                            canonical_route_contributions;
                        gather_params.seq_len = total_tokens;
                        gather_params.top_k = config_.moe.top_k;
                        gather_params.d_model = config_.d_model;
                        gather_params.root_participant =
                            canonical_route_root_participant;
                        gather_params.stage_name = ar_name;
                        gather_params.packed_route_records_buffer_id =
                            allreduce_buffer_id;
                        collective_stage =
                            ComputeStageFactory::createMoECanonicalRouteGather(
                                gather_params);
                    }
                    else if (canonical_publication_lowering.usesRootedCollective())
                    {
                        canonical_route_root_participant =
                            continuationRootParticipant(*overlay_plan);
                        if (!local_tp_ctx ||
                            config_.tp_device_idx < 0 ||
                            config_.tp_device_idx >= local_tp_ctx->degree() ||
                            canonical_route_root_participant < 0 ||
                            canonical_route_root_participant >=
                                local_tp_ctx->degree())
                        {
                            throw std::runtime_error(
                                "Qwen35 MoE canonical rooted route publication "
                                "requires a valid homogeneous LocalTP participant "
                                "and root for layer " +
                                std::to_string(layer_idx));
                        }

                        TPLocalRootedCollectiveStage::Params rooted_params;
                        rooted_params.device_id = device;
                        rooted_params.tp_ctx = local_tp_ctx;
                        rooted_params.tensor = allreduce_buffer;
                        rooted_params.count = allreduce_count;
                        rooted_params.dtype = CollectiveDataType::FLOAT32;
                        rooted_params.operation =
                            TPLocalRootedCollectiveOperation::ReduceSum;
                        rooted_params.root_device_index =
                            canonical_route_root_participant;
                        rooted_params.participant_device_index =
                            config_.tp_device_idx;
                        rooted_params.stage_name = ar_name;
                        rooted_params.tensor_buffer_id = allreduce_buffer_id;
                        rooted_params.sideband_workspace_bindings =
                            std::move(rebalance_sidebands);
                        collective_stage =
                            ComputeStageFactory::createTPLocalRootedCollective(
                                rooted_params);
                    }
                    else
                    {
                        collective_stage = createTPAllreduceStage(
                            allreduce_buffer,
                            allreduce_count,
                            device,
                            layer_idx,
                            /*is_attention=*/false,
                            ar_name,
                            allreduce_buffer_id,
                            std::move(rebalance_sidebands));
                    }
                    if (!collective_stage)
                    {
                        throw std::runtime_error(
                            "Qwen35 MoE graph could not create the required "
                            "LocalTP routed-expert collective for layer " +
                            std::to_string(layer_idx));
                    }
                    graph.addNode(ar_name, std::move(collective_stage), device);
                    /*
                     * The rooted transaction always consumes routed slots.
                     * Keep that producer edge explicit even when the rank-bank
                     * policy later adds a shared publisher and when rebalance
                     * state is piggybacked on the same collective. Those nodes
                     * are additional producers, never substitutes for routed
                     * evidence. The collective therefore cannot race either
                     * branch as the graph evolves.
                     */
                    graph.addDependency(
                        ar_name,
                        prefix + "moe_expert_ffn_overlay_fast");
                    if (!graph_rebalance_collect_node.empty())
                    {
                        graph.addDependency(
                            ar_name,
                            graph_rebalance_collect_node);
                    }
                    if (graph_rebalance_plan_after_sideband_params.has_value() &&
                        !graph_rebalance_plan_after_sideband_node.empty())
                    {
                        graph.addNode(
                            graph_rebalance_plan_after_sideband_node,
                            ComputeStageFactory::createMoEDeviceRebalance(
                                *graph_rebalance_plan_after_sideband_params),
                            device);
                        graph.addDependency(
                            graph_rebalance_plan_after_sideband_node,
                            ar_name);
                        ffn_terminal =
                            graph_rebalance_plan_after_sideband_node;
                        graph_rebalance_plan_after_sideband_node.clear();
                        graph_rebalance_plan_after_sideband_params.reset();
                    }
                    else
                    {
                        ffn_terminal = ar_name;
                    }
                    maybeAddGraphRebalancePayloadStageAfterSideband(
                        ar_name,
                        ffn_terminal);

                    if (canonical_local_tp_route_publication &&
                        canonical_publication_lowering.usesRankBanks())
                    {
                        /*
                         * Shared publication is created after its FFN branch.
                         * Retain the complete rooted-transaction identity now
                         * so later lowering can add the missing producer edge,
                         * root finalizer, and final-row broadcast without
                         * rediscovering topology or collective policy.
                         */
                        canonical_publication_lowering = {
                            .policy = MoEParticipantPublicationPolicy::
                                CanonicalRootedRankBanks,
                            .root_participant =
                                canonical_route_root_participant,
                            .participant_count =
                                expert_params.participant_count,
                            .routed_producer =
                                prefix + "moe_expert_ffn_overlay_fast",
                            .rooted_reduce_node = ar_name,
                            .post_collective_terminal = ffn_terminal,
                        };
                    }
                    else if (canonical_publication_lowering.
                                 usesRootedCollective())
                    {
                        MoECanonicalRouteReduceStage::Params reduce_params;
                        reduce_params.device_id = device;
                        reduce_params.canonical_route_contributions =
                            canonical_route_contributions;
                        reduce_params.output = moe_output;
                        reduce_params.seq_len = total_tokens;
                        reduce_params.top_k = config_.moe.top_k;
                        reduce_params.d_model = config_.d_model;
                        reduce_params.canonical_route_arithmetic =
                            MoECanonicalRouteArithmeticPolicy::
                                PreweightedContributionThenOrderedAdd;
                        reduce_params.canonical_route_layout =
                            MoECanonicalRoutePublicationLayout::
                                DenseOriginalRouteSlots;
                        reduce_params.reduction_role =
                            config_.tp_device_idx ==
                                    canonical_route_root_participant
                                ? MoECanonicalRouteReductionRole::RootOwner
                                : MoECanonicalRouteReductionRole::
                                      NonRootParticipant;
                        reduce_params.canonical_route_contributions_buffer_id =
                            allreduce_buffer_id;
                        reduce_params.output_buffer_id =
                            buffers.idFor(BufferId::MOE_COMBINED_OUTPUT);

                        const std::string reduce_name =
                            prefix + "moe_canonical_routes_reduce";
                        graph.addNode(
                            reduce_name,
                            ComputeStageFactory::createMoECanonicalRouteReduce(
                                reduce_params),
                            device);
                        graph.addDependency(reduce_name, ffn_terminal);

                        TPLocalRootedCollectiveStage::Params broadcast_params;
                        broadcast_params.device_id = device;
                        broadcast_params.tp_ctx = local_tp_ctx;
                        broadcast_params.tensor = moe_output;
                        broadcast_params.count =
                            static_cast<size_t>(total_tokens) *
                            static_cast<size_t>(config_.d_model);
                        broadcast_params.dtype =
                            CollectiveDataType::FLOAT32;
                        broadcast_params.operation =
                            TPLocalRootedCollectiveOperation::Broadcast;
                        broadcast_params.root_device_index =
                            canonical_route_root_participant;
                        broadcast_params.participant_device_index =
                            config_.tp_device_idx;
                        broadcast_params.stage_name =
                            prefix + "moe_canonical_routes_broadcast";
                        broadcast_params.tensor_buffer_id =
                            buffers.idFor(BufferId::MOE_COMBINED_OUTPUT);
                        const std::string broadcast_name =
                            broadcast_params.stage_name;
                        graph.addNode(
                            broadcast_name,
                            ComputeStageFactory::
                                createTPLocalRootedCollective(
                                    broadcast_params),
                            device);
                        graph.addDependency(broadcast_name, reduce_name);
                        ffn_terminal = broadcast_name;
                    }
                    else if (canonical_publication_lowering.
                                 usesPackedRouteGather())
                    {
                        /*
                         * The gather transported only locally computed indexed
                         * rows. The fixed root reconstructs original slot order
                         * and owns the one serial-equivalent FMA fold; peers
                         * remain declarative no-ops until compact broadcast.
                         */
                        MoECanonicalRouteReduceStage::Params reduce_params;
                        reduce_params.device_id = device;
                        reduce_params.canonical_route_contributions =
                            canonical_route_contributions;
                        reduce_params.routing_weights = routing_weights;
                        reduce_params.output = moe_output;
                        reduce_params.seq_len = total_tokens;
                        reduce_params.top_k = config_.moe.top_k;
                        reduce_params.d_model = config_.d_model;
                        reduce_params.canonical_route_arithmetic =
                            MoECanonicalRouteArithmeticPolicy::
                                UnweightedExpertRowThenOrderedFMA;
                        reduce_params.canonical_route_layout =
                            MoECanonicalRoutePublicationLayout::
                                PackedIndexedRouteRows;
                        reduce_params.reduction_role =
                            config_.tp_device_idx ==
                                    canonical_route_root_participant
                                ? MoECanonicalRouteReductionRole::RootOwner
                                : MoECanonicalRouteReductionRole::
                                      NonRootParticipant;
                        reduce_params.canonical_route_contributions_buffer_id =
                            allreduce_buffer_id;
                        reduce_params.routing_weights_buffer_id =
                            buffers.idFor(BufferId::MOE_EXPERT_WEIGHTS);
                        reduce_params.output_buffer_id =
                            buffers.idFor(BufferId::MOE_COMBINED_OUTPUT);

                        const std::string reduce_name =
                            prefix + "moe_canonical_routes_ordered_fma";
                        graph.addNode(
                            reduce_name,
                            ComputeStageFactory::createMoECanonicalRouteReduce(
                                reduce_params),
                            device);
                        graph.addDependency(reduce_name, ffn_terminal);

                        MoECanonicalOutputBroadcastStage::Params
                            broadcast_params;
                        broadcast_params.device_id = device;
                        broadcast_params.tp_ctx = config_.tp_ctx;
                        broadcast_params.output = moe_output;
                        broadcast_params.seq_len = total_tokens;
                        broadcast_params.d_model = config_.d_model;
                        broadcast_params.root_participant =
                            canonical_route_root_participant;
                        broadcast_params.stage_name =
                            prefix + "moe_canonical_routes_broadcast";
                        broadcast_params.output_buffer_id =
                            buffers.idFor(BufferId::MOE_COMBINED_OUTPUT);
                        const std::string broadcast_name =
                            broadcast_params.stage_name;
                        graph.addNode(
                            broadcast_name,
                            ComputeStageFactory::
                                createMoECanonicalOutputBroadcast(
                                    broadcast_params),
                            device);
                        graph.addDependency(broadcast_name, reduce_name);
                        ffn_terminal = broadcast_name;
                    }
                }

                LOG_TRACE("[Qwen35MoEGraph] Layer " << layer_idx
                                                    << " using LocalTP routed-expert fast path on "
                                                    << device.to_string()
                                                    << " participant=" << local_participant
                                                    << " domain=" << domain_name
                                                    << " compute="
                                                    << routedExpertComputePolicyToString(
                                                           config_.moe.routed_compute_policy)
                                                    << " row_execution="
                                                    << routedExpertRowExecutionPolicyToString(
                                                           routed_row_execution_policy)
                                                    << " publication="
                                                    << moeParticipantPublicationPolicyToString(
                                                           canonical_publication_lowering.policy)
                                                    << " phase="
                                                    << (local_tp_fast_tier
                                                            ? routedExpertPhasePolicyToString(
                                                                  expertDomainForTier(
                                                                      *overlay_plan,
                                                                      *local_tp_fast_tier)
                                                                      ->routed_phase_policy)
                                                            : "uniform"));
            }
            else if (use_expert_overlay)
            {
                auto owner_map_lifetime = std::make_shared<MoEExpertOwnerMap>(
                    MoEExpertOwnerMap::build(*overlay_plan));
                const int continuation_root_participant =
                    continuationRootParticipant(*overlay_plan);
                const int participant_count =
                    participantCountForGraphNativeOverlay(
                        *owner_map_lifetime,
                        continuation_root_participant);
                const bool distributed_overlay =
                    distributed_expert_overlay;
                std::vector<int> local_overlay_participants;
                if (distributed_overlay)
                {
                    const int current_world_rank =
                        config_.moe.overlay_mpi_ctx->rank();
                    for (const auto &participant :
                         owner_map_lifetime->participants())
                    {
                        if (participant.world_rank_known &&
                            participant.world_rank == current_world_rank)
                        {
                            local_overlay_participants.push_back(
                                participant.participant_id);
                        }
                    }
                }
                else
                {
                    for (const auto &participant :
                         owner_map_lifetime->participants())
                    {
                        local_overlay_participants.push_back(
                            participant.participant_id);
                    }
                }
                std::sort(
                    local_overlay_participants.begin(),
                    local_overlay_participants.end());
                local_overlay_participants.erase(
                    std::unique(
                        local_overlay_participants.begin(),
                        local_overlay_participants.end()),
                    local_overlay_participants.end());
                const auto owns_overlay_participant =
                    [&](int participant_id)
                {
                    return std::binary_search(
                        local_overlay_participants.begin(),
                        local_overlay_participants.end(),
                        participant_id);
                };

                const int continuation_tier_index =
                    routed_continuation_topology.tier_index;
                const RoutedExpertDomain *continuation_expert_domain =
                    routed_continuation_topology.domain;
                const int graph_local_continuation_participant =
                    routed_continuation_topology
                        .graph_local_participant;
                /*
                 * Dynamic mapped continuations use the same complete captured
                 * sparse endpoint graph as Static. The mutable choice is not a
                 * host-side graph branch: every MoE runtime table is bound to
                 * DeviceMoEOverlayEpochArena, whose acquire kernel reads the
                 * topology-wide mapped admission epoch and pins one immutable
                 * bank for the complete request transaction. Maintenance may
                 * therefore prepare and publish the inactive bank on its own
                 * streams without changing this graph's topology or embedded
                 * descriptor addresses.
                 */
                const bool owns_continuation_root =
                    captured_overlay_continuation
                        ? graph_local_continuation_participant ==
                              continuation_root_participant
                        : owns_overlay_participant(
                              continuation_root_participant);
                const DistributedSparseGraphContract sparse_graph_contract =
                    resolveDistributedSparseGraphContract(
                        distributed_overlay,
                        captured_overlay_continuation,
                        owns_continuation_root);
                routed_overlay_has_distributed_sparse_protocol =
                    distributed_overlay;
                if (config_.moe.expert_overlay_residency_authority &&
                    !config_.moe.expert_overlay_participant_residency)
                {
                    throw std::runtime_error(
                        "Qwen35 MoE production ExpertOverlay graph has a "
                        "global residency authority but no process-local "
                        "epoch-indexed prepared banks");
                }

                std::string captured_local_expert_compute_node;
                std::string captured_local_expert_terminal;
                std::string captured_overlay_ticket_wave;
                std::string captured_overlay_outbound_wave;
                std::string captured_overlay_inbound_wave;
                int continuation_root_tp_index = -1;
                if (captured_overlay_continuation)
                {
                    /*
                     * Every continuation GPU first records the common router
                     * prefix. The logical root publishes all remote packets,
                     * then its continuation-local expert branch runs in the
                     * same retained device parent while follower devices work
                     * concurrently. Rank-local TP continuations additionally
                     * reduce their fixed route-slot banks and broadcast the
                     * completed row; a single-device continuation needs
                     * neither collective and retains its local output in
                     * MOE_COMBINED_OUTPUT for the ordered remote folds.
                     */
                    captured_overlay_ticket_wave =
                        prefix + "moe_overlay_ticket_capture";
                    captured_overlay_outbound_wave =
                        prefix + "moe_overlay_outbound_capture";
                    captured_overlay_inbound_wave =
                        prefix + "moe_overlay_inbound_capture";
                    graph.setGraphCaptureWaveContract(
                        prefix + "moe_routing",
                        GraphCaptureWaveContract{
                            .identity = captured_overlay_ticket_wave,
                        });
                    if (!canonical_route_contributions)
                    {
                        throw std::runtime_error(
                            "Qwen35 MoE captured distributed continuation requires canonical route storage");
                    }
                    const auto *local_participant =
                        owner_map_lifetime->participantForId(
                            graph_local_continuation_participant);
                    const auto *root_participant =
                        owner_map_lifetime->participantForId(
                            continuation_root_participant);
                    if (!local_participant || !root_participant ||
                        local_participant->domain_name !=
                            overlay_plan->continuation_domain ||
                        root_participant->domain_name !=
                            overlay_plan->continuation_domain)
                    {
                        throw std::runtime_error(
                            "Qwen35 MoE captured distributed continuation has an unaligned logical participant mapping");
                    }
                    if (captured_local_tp_continuation)
                    {
                        if (!local_tp_ctx ||
                            local_participant->domain_participant_index !=
                                config_.tp_device_idx ||
                            root_participant->domain_participant_index < 0 ||
                            root_participant->domain_participant_index >=
                                local_tp_ctx->degree())
                        {
                            throw std::runtime_error(
                                "Qwen35 MoE captured distributed LocalTP continuation has an unaligned graph-device/participant mapping");
                        }
                        continuation_root_tp_index =
                            root_participant->domain_participant_index;
                    }
                    else if (graph_local_continuation_participant !=
                                 continuation_root_participant ||
                             !continuation_expert_domain ||
                             continuation_expert_domain->scope !=
                                 ExecutionDomainScope::SINGLE ||
                             continuation_expert_domain->participants.size() !=
                                 1u ||
                             local_participant->domain_participant_index != 0)
                    {
                        throw std::runtime_error(
                            "Qwen35 MoE captured single-device continuation has an invalid root identity");
                    }

                    auto participant_mask =
                        owner_map_lifetime->expertMaskForParticipant(
                            layer_idx,
                            graph_local_continuation_participant,
                            config_.moe.num_experts);
                    if (!hasActiveExpertMask(participant_mask))
                    {
                        throw std::runtime_error(
                            "Qwen35 MoE captured distributed continuation produced an empty graph-local expert mask");
                    }

                    auto local_params = makeExpertParams(
                        moe_output,
                        buffers.idFor(BufferId::MOE_COMBINED_OUTPUT),
                        std::move(participant_mask),
                        device);
                    local_params.my_socket_id =
                        local_participant->domain_participant_index;
                    local_params.participant_count =
                        captured_local_tp_continuation
                            ? local_tp_ctx->degree()
                            : 1;
                    local_params.require_device_routing_tensor_decode =
                        device.is_gpu();
                    /* Every captured heterogeneous continuation publishes its
                     * graph-local routes in original router slots. LocalTP
                     * transports those slots between continuation devices;
                     * a single-device continuation folds them locally before
                     * mapped remote returns are joined. Keeping one producer
                     * contract makes the local arithmetic and diagnostics
                     * independent of transport topology. */
                    local_params.canonical_route_contributions =
                        canonical_route_contributions;
                    local_params.canonical_route_arithmetic =
                        MoECanonicalRouteArithmeticPolicy::
                            PreweightedContributionThenOrderedAdd;
                    local_params.canonical_route_layout =
                        MoECanonicalRoutePublicationLayout::
                            DenseOriginalRouteSlots;
                    local_params.canonical_route_contributions_buffer_id =
                        buffers.idFor(
                            BufferId::MOE_CANONICAL_ROUTE_CONTRIBUTIONS);

                    if (!prepareExpertParams(
                            local_params,
                            device,
                            "captured distributed continuation participant " +
                                std::to_string(
                                    graph_local_continuation_participant),
                            overlay_plan->continuation_domain))
                    {
                        throw std::runtime_error(
                            "Qwen35 MoE could not prepare captured continuation-local experts for layer " +
                            std::to_string(layer_idx) + " on " +
                            device.to_string());
                    }

                    if (captured_distributed_overlay_runtime_table)
                    {
                        const auto domain_local_owners =
                            domainLocalOwnerParticipantsFromMap(
                                *owner_map_lifetime,
                                layer_idx,
                                config_.moe.num_experts,
                                overlay_plan->continuation_domain);
                        const auto overlay_route_participants =
                            ownerParticipantsFromMap(
                                *owner_map_lifetime,
                                layer_idx,
                                config_.moe.num_experts);
                        if (!initializeMaskedLocalDecodeRuntimeTable(
                                moe_runtime_table,
                                layer_idx,
                                config_.moe.num_experts,
                                config_.moe.top_k,
                                config_.d_model,
                                expert_intermediate,
                                local_params.expert_mask,
                                local_params.my_socket_id,
                                local_params.participant_count,
                                domain_local_owners,
                                overlay_route_participants,
                                local_params.prepared_gate_gemm,
                                local_params.prepared_up_gemm,
                                local_params.prepared_down_gemm,
                                device_state_publication_stream,
                                /*allow_existing_dynamic_bank=*/true,
                                "captured distributed continuation decode runtime bank"))
                        {
                            throw std::runtime_error(
                                "Qwen35 MoE could not initialize the captured continuation-local decode runtime table for layer " +
                                std::to_string(layer_idx) + " on " +
                                device.to_string());
                        }
                        if (total_tokens > 1 &&
                            captured_overlay_route_ledger_uses_grouped_publication)
                        {
                            local_params.use_runtime_row_grouping = true;
                        }
                    }

                    registerInitialOverlayParticipantBank(
                        *owner_map_lifetime,
                        graph_local_continuation_participant,
                        local_params,
                        "Qwen35 MoE captured distributed continuation");

                    captured_local_expert_compute_node =
                        prefix +
                        "moe_expert_ffn_overlay_continuation_local";
                    graph.addNode(
                        captured_local_expert_compute_node,
                        ComputeStageFactory::createMoEExpertCompute(
                            local_params),
                        device);
                    graph.addDependency(
                        captured_local_expert_compute_node,
                        prefix + "moe_routing");
                    GraphCaptureWaveContract local_capture_wave{
                        .identity = captured_overlay_outbound_wave,
                    };
                    graph.setGraphCaptureWaveContract(
                        captured_local_expert_compute_node,
                        std::move(local_capture_wave));

                    captured_local_expert_terminal =
                        captured_local_expert_compute_node;
                    /* Local and remote experts publish disjoint original route
                     * slots into one canonical bank. Do not reduce the local
                     * prefix here: the sole ordered reducer is installed only
                     * after every mapped return has materialized its slots. */
                }

                /*
                 * CPU least-loaded prefill is a transient child of the one
                 * durable ExpertOverlay epoch.  The state is graph-owned and
                 * shared by the begin planner, root descriptor, the one local
                 * sparse endpoint on each rank, and restore.  No child owner
                 * map is published: the durable lease remains the only
                 * placement authority until every sparse return completes.
                 */
                std::shared_ptr<CPUCurrentBatchLLEPTransactionState>
                    cpu_llep_state;
                if (cpu_llep_prefill_transport_supported)
                {
                    cpu_llep_state =
                        std::make_shared<
                            CPUCurrentBatchLLEPTransactionState>();
                }
                const auto host_dispatch_lease_owner =
                    cpu_llep_state
                        ? MoEOverlayHostDispatchLeaseOwner::CurrentBatchLLEP
                        : config_.moe.expert_overlay_residency_authority
                              ? MoEOverlayHostDispatchLeaseOwner::
                                    FinalSparseReturn
                              : MoEOverlayHostDispatchLeaseOwner::None;
                MoELocalExpertStage *cpu_llep_local_consumer = nullptr;
                std::vector<std::string> cpu_llep_sparse_dispatch_nodes;

                auto dispatch_output_lifetime = std::make_shared<MoEExpertDispatchOutput>();
                std::shared_ptr<MoEOverlayDispatchTicketStorage>
                    dispatch_ticket_storage;
                std::string dispatch_ticket_publish_name;
                const std::string dispatch_name =
                    prefix + "moe_expert_dispatch";
                std::string dispatch_dependency =
                    prefix + "moe_routing";
                bool host_dispatch_path_materialized = false;
                const auto ensure_host_dispatch_path = [&]()
                {
                    if (host_dispatch_path_materialized)
                        return;
                    if (sparse_graph_contract.ownsDispatchAuthority() &&
                        device.is_gpu())
                    {
                        dispatch_ticket_storage =
                            std::make_shared<
                                MoEOverlayDispatchTicketStorage>();
                        dispatch_ticket_storage->bindFixedCapacity(
                            layer_idx,
                            total_tokens,
                            config_.moe.top_k,
                            config_.d_model,
                            device,
                            /*workspace_generation=*/1,
                            mappedOverlayTicketArenaForDevice(device));
                        dispatch_output_lifetime->ticket_lifetime =
                            dispatch_ticket_storage;

                        const std::string ticket_publish_dependency =
                            prefix + "moe_routing";

                        MoEOverlayTicketPublishStage::Params ticket_params;
                        ticket_params.device_id = device;
                        ticket_params.hidden = buffers.normalized;
                        ticket_params.routing_indices = routing_indices;
                        ticket_params.routing_weights = routing_weights;
                        ticket_params.hidden_buffer_id =
                            buffers.idFor(BufferId::NORMALIZED);
                        ticket_params.routing_indices_buffer_id =
                            buffers.idFor(BufferId::MOE_EXPERT_INDICES);
                        ticket_params.routing_weights_buffer_id =
                            buffers.idFor(BufferId::MOE_EXPERT_WEIGHTS);
                        /*
                         * A decode graph always owns one activation row even
                         * though sequence_lengths_device contains the growing
                         * KV position.  Only a padded multi-row prefill bucket
                         * needs that scalar to distinguish real prompt rows
                         * from physical padding.
                         */
                        ticket_params.active_row_count_device =
                            batch_size == 1 && total_tokens > 1
                                ? sequence_lengths_device
                                : nullptr;
                        ticket_params.layer_idx = layer_idx;
                        ticket_params.bucket_rows = total_tokens;
                        ticket_params.top_k = config_.moe.top_k;
                        ticket_params.d_model = config_.d_model;
                        ticket_params.ticket_storage = dispatch_ticket_storage;

                        dispatch_ticket_publish_name =
                            prefix + "moe_overlay_ticket_publish";
                        graph.addNode(
                            dispatch_ticket_publish_name,
                            ComputeStageFactory::createMoEOverlayTicketPublish(
                                ticket_params),
                            device);
                        graph.addDependency(
                            dispatch_ticket_publish_name,
                            ticket_publish_dependency);
                        if (captured_overlay_continuation)
                        {
                            graph.setGraphCaptureWaveContract(
                                dispatch_ticket_publish_name,
                                GraphCaptureWaveContract{
                                    .identity =
                                        captured_overlay_ticket_wave,
                                });
                        }
                    }

                    MoEExpertDispatchStage::Params dispatch_params;
                    dispatch_params.device_id = DeviceId::cpu();
                    dispatch_params.ticket_storage = dispatch_ticket_storage;
                    if (!dispatch_ticket_storage)
                    {
                        dispatch_params.routing_indices = routing_indices;
                        dispatch_params.routing_weights = routing_weights;
                        dispatch_params.hidden = buffers.normalized;
                        dispatch_params.routing_indices_buffer_id =
                            buffers.idFor(BufferId::MOE_EXPERT_INDICES);
                        dispatch_params.routing_weights_buffer_id =
                            buffers.idFor(BufferId::MOE_EXPERT_WEIGHTS);
                        dispatch_params.hidden_buffer_id =
                            buffers.idFor(BufferId::NORMALIZED);
                    }
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
                    dispatch_params.owner_map = owner_map_lifetime;
                    dispatch_params.residency_authority =
                        config_.moe.expert_overlay_residency_authority;
                    dispatch_params.cpu_current_batch_llep_state =
                        cpu_llep_state;
                    if (dispatch_ticket_storage &&
                        config_.moe.decode_histogram &&
                        !mtp_sidecar_context &&
                        !grouped_main_verifier_layer)
                    {
                    /*
                     * The manual ticket boundary already owns the exact host
                     * route prefix. Publish ordinary prefill/decode demand
                     * there; grouped verification must wait for accepted-state
                     * publication and is deliberately excluded here.
                     */
                        dispatch_params.routing_evidence_publication =
                            MoEExpertDispatchStage::RoutingEvidencePublication{
                                .histogram = config_.moe.decode_histogram,
                                .source = ordinary_prefill_graph
                                              ? ExpertHistogramSource::PrefillChunk
                                              : ExpertHistogramSource::DecodeToken,
                            };
                    }
                    dispatch_params.output_lifetime = dispatch_output_lifetime;

                    if (sparse_graph_contract.ownsDispatchAuthority())
                    {
                        graph.addNode(
                            dispatch_name,
                            ComputeStageFactory::createMoEExpertDispatch(
                                dispatch_params),
                            DeviceId::cpu());
                        graph.addDependency(
                            dispatch_name,
                            dispatch_ticket_publish_name.empty()
                                ? prefix + "moe_routing"
                                : dispatch_ticket_publish_name);
                        dispatch_dependency = dispatch_name;
                        if (captured_overlay_continuation &&
                            !captured_local_expert_compute_node.empty())
                        {
                            /*
                             * Ticket publication is the immutable fork point
                             * for continuation-local GPU work. The GPU branch
                             * must never wait for host parsing: it consumes the
                             * already-published device bytes directly.
                             */
                            graph.addDependency(
                                captured_local_expert_compute_node,
                                dispatch_ticket_publish_name.empty()
                                    ? dispatch_name
                                    : dispatch_ticket_publish_name);

                            if (!dispatch_ticket_publish_name.empty() &&
                                requiresHeterogeneousTicketSegmentation(
                                    graph.nativeCaptureEnvelope()))
                            {
                                /*
                                 * A retained heterogeneous transaction has one
                                 * manual CPU unit per layer. Materializing the
                                 * host route table before the captured local
                                 * expert producer created a second host gap:
                                 * captured ticket -> host parse -> captured
                                 * local expert -> host sparse transaction.
                                 *
                                 * Make host parsing consume the completed local
                                 * producer instead. The downstream sparse CPU
                                 * stages already depend on this dispatch node,
                                 * so all host work becomes one adjacent manual
                                 * unit after the typed pre-CPU cutpoint. This
                                 * preserves device/host independence (the GPU
                                 * never waits on the host), removes a redundant
                                 * retained launch, and gives LocalTP authority
                                 * and followers one symmetric capture boundary.
                                 */
                                graph.addDependency(
                                    dispatch_name,
                                    captured_local_expert_compute_node);
                            }
                        }
                    }
                    host_dispatch_path_materialized = true;
                };

                struct OverlayTierTargets
                {
                    std::vector<int> participants;
                    bool graph_local_apportioned = false;
                };
                const auto resolve_overlay_tier_targets =
                    [&](size_t tier_index) -> OverlayTierTargets
                {
                    const auto &tier =
                        overlay_plan->routed_tiers[tier_index];
                    OverlayTierTargets result;
                    /*
                     * A heterogeneous GPU continuation executes its own tier
                     * directly inside each captured device graph. Only the
                     * exact logical root enters the rank-level sparse protocol,
                     * and that protocol addresses remote tiers only. This also
                     * prevents LocalTP siblings from submitting duplicate
                     * cross-rank collectives.
                     */
                    if (captured_overlay_continuation &&
                        static_cast<int>(tier_index) ==
                            continuation_tier_index)
                    {
                        return result;
                    }
                    if (captured_overlay_continuation &&
                        distributed_overlay &&
                        !sparse_graph_contract
                             .participatesInRankBatchProtocol())
                    {
                        return result;
                    }
                    result.participants =
                        owner_map_lifetime->participantIdsForTier(
                            static_cast<int>(tier_index));
                    const bool local_tp_apportioned =
                        isLocalTPExpertIdApportionedTier(
                            *overlay_plan,
                            tier);
                    const int graph_local_participant =
                        local_tp_apportioned
                            ? participantIdForTierDevice(
                                  *owner_map_lifetime,
                                  static_cast<int>(tier_index),
                                  device)
                            : -1;
                    result.graph_local_apportioned =
                        graph_local_participant >= 0 &&
                        (!distributed_overlay ||
                         owns_overlay_participant(
                             graph_local_participant));
                    if (result.graph_local_apportioned)
                        result.participants = {graph_local_participant};
                    return result;
                };

                /*
                 * A same-node distributed GPU continuation lowers directly to
                 * mapped activation epochs only when every remote participant
                 * in this graph family is physically node-local. Mixing a
                 * portable inter-node message into the retained mapped parent
                 * would reintroduce a host-owned transaction boundary, so a
                 * mixed physical topology remains one explicit MPI protocol.
                 * Participant lanes come from the frozen topology rather than
                 * the initial expert masks: an initially empty endpoint must
                 * remain addressable after a later placement-bank migration.
                 */
                const size_t mapped_activation_graph_family_ordinal =
                    mtp_sidecar_context
                        ? static_cast<size_t>(
                              std::max(0, mtp_depth_idx) + 1)
                        : 0u;
                std::vector<int> mapped_activation_remote_participants;
                std::vector<int> mapped_activation_local_participants;
                std::vector<std::string>
                    mapped_activation_dispatch_wave_identities;
                std::vector<std::string>
                    mapped_activation_return_wave_identities;
                const auto mapped_dispatch_wave_identity =
                    [&](int participant)
                {
                    return prefix +
                           "moe_overlay_mapped_dispatch_capture_family" +
                           std::to_string(
                               mapped_activation_graph_family_ordinal) +
                           "_p" + std::to_string(participant);
                };
                const auto mapped_return_wave_identity =
                    [&](int participant)
                {
                    return prefix +
                           "moe_overlay_mapped_return_capture_family" +
                           std::to_string(
                               mapped_activation_graph_family_ordinal) +
                           "_p" + std::to_string(participant);
                };
                bool use_mapped_activation_parent = false;
                MoEOverlayRoutePlacementDeviceBinding
                    pinned_route_placement;
                MoEDomainRouteAssignmentLedger
                    pinned_domain_route_assignment{};
                MoERuntimeRouteWeightBinding
                    pinned_runtime_route_weights{};
                if (captured_overlay_continuation &&
                    device.is_gpu())
                {
                    const auto *const source_descriptor =
                        owner_map_lifetime->participantForId(
                            continuation_root_participant);
                    if (!source_descriptor ||
                        !source_descriptor->world_rank_known ||
                        source_descriptor->world_rank !=
                            config_.moe.overlay_mpi_ctx->rank())
                    {
                        throw std::logic_error(
                            "Qwen35 MoE mapped activation parent cannot resolve its continuation rank");
                    }

                    bool saw_remote_participant = false;
                    bool all_remote_participants_are_node_local = true;
                    for (size_t tier_index = 0;
                         tier_index < overlay_plan->routed_tiers.size();
                         ++tier_index)
                    {
                        /* The complete continuation tier is already represented
                         * by symmetric graph-local branches. Inspect every
                         * other topology-declared endpoint even when its
                         * initial mask is empty so migration cannot require a
                         * recapture. */
                        if (static_cast<int>(tier_index) ==
                            continuation_tier_index)
                        {
                            continue;
                        }
                        auto tier_participants =
                            owner_map_lifetime->participantIdsForTier(
                                static_cast<int>(tier_index));
                        std::sort(
                            tier_participants.begin(),
                            tier_participants.end());
                        for (const int target_participant :
                             tier_participants)
                        {
                            const auto *const target_descriptor =
                                owner_map_lifetime->participantForId(
                                    target_participant);
                            if (!target_descriptor ||
                                !target_descriptor->world_rank_known)
                            {
                                throw std::logic_error(
                                    "Qwen35 MoE mapped activation topology contains an unresolved endpoint");
                            }
                            if (target_descriptor->world_rank ==
                                source_descriptor->world_rank)
                            {
                                /*
                                 * A NodeTP domain can place one lower-tier CPU
                                 * endpoint beside the continuation GPU. That
                                 * edge is graph-local and must never be
                                 * represented by a self-MPI or mapped-rank
                                 * channel. It is lowered through the fixed
                                 * direct sparse ticket path below.
                                 */
                                mapped_activation_local_participants.push_back(
                                    target_participant);
                                continue;
                            }
                            saw_remote_participant = true;
                            mapped_activation_remote_participants.push_back(
                                target_participant);
                            if (resolveMoEOverlayRankBatchTransportKind(
                                    *config_.moe.overlay_mpi_ctx,
                                    source_descriptor->world_rank,
                                    target_descriptor->world_rank) !=
                                MoEOverlayRankBatchTransportKind::
                                    NodeLocalSharedRows)
                            {
                                all_remote_participants_are_node_local = false;
                            }
                        }
                    }
                    const bool mapped_activation_topology =
                        saw_remote_participant &&
                        all_remote_participants_are_node_local;
                    if (mapped_activation_topology)
                    {
                        /*
                         * A purely GPU/mapped endpoint transaction remains one
                         * indivisible native executable. A colocated CPU
                         * endpoint introduces one intentional fixed-ticket
                         * boundary: captured GPU publication, manual CPU sparse
                         * work, then captured GPU ingress. Declare that
                         * lifecycle explicitly so replay policy cannot infer it
                         * from mutable placement masks or broad topology flags.
                         */
                        if (mapped_activation_local_participants.empty())
                        {
                            graph.setNativeCaptureEnvelope(
                                GraphNativeCaptureEnvelope::
                                    DeviceOwnedTimelineTransaction);
                        }
                        else
                        {
                            /*
                             * The CPU ticket cuts the continuation domain's
                             * native timeline, not just the logical root's
                             * graph. The root owns the manual ticket boundary;
                             * every LocalTP sibling follows the same segmented
                             * wave schedule without executing host work.
                             */
                            graph.setNativeCaptureEnvelope(
                                sparse_graph_contract
                                        .ownsDispatchAuthority()
                                    ? GraphNativeCaptureEnvelope::
                                          HeterogeneousTicketAuthorityTransaction
                                    : GraphNativeCaptureEnvelope::
                                          HeterogeneousTicketFollowerTransaction);
                        }
                        mapped_activation_dispatch_wave_identities.reserve(
                            mapped_activation_remote_participants.size());
                        mapped_activation_return_wave_identities.reserve(
                            mapped_activation_remote_participants.size());
                        for (const int participant :
                             mapped_activation_remote_participants)
                        {
                            mapped_activation_dispatch_wave_identities.push_back(
                                mapped_dispatch_wave_identity(participant));
                            mapped_activation_return_wave_identities.push_back(
                                mapped_return_wave_identity(participant));
                        }
                    }
                    else if (!mapped_activation_local_participants.empty())
                    {
                        /* A one-rank GPU+CPU topology still crosses the same
                         * explicit captured/manual/captured boundary. Its
                         * lifecycle is heterogeneous even though no MPI peer
                         * exists, so make segmentation a declared graph fact. */
                        graph.setNativeCaptureEnvelope(
                            GraphNativeCaptureEnvelope::
                                HeterogeneousTicketAuthorityTransaction);
                    }
                    use_mapped_activation_parent =
                        sparse_graph_contract.ownsDispatchAuthority() &&
                        sparse_graph_contract
                            .participatesInRankBatchProtocol() &&
                        mapped_activation_topology;
                    if (sparse_graph_contract.ownsDispatchAuthority())
                    {
                        /*
                         * Transport choice does not own routing truth. Mapped
                         * GPU lanes, a colocated CPU ticket, and portable MPI
                         * packets all consume the same device-authored route
                         * ledger and placement epoch. Bind those values for
                         * every continuation authority so the final reducer
                         * can publish exact evidence even when this topology
                         * has no remote mapped GPU participant.
                         */
                        if (!moe_runtime_table)
                        {
                            throw std::runtime_error(
                                "Qwen35 MoE captured overlay continuation requires the canonical device runtime placement table");
                        }
                        pinned_route_placement =
                            moe_runtime_table->overlayRoutePlacementBinding(
                                layer_idx);
                        if (!pinned_route_placement.valid())
                        {
                            throw std::runtime_error(
                                "Qwen35 MoE captured overlay continuation received an incomplete placement-bank binding");
                        }
                        const auto &runtime_layer =
                            moe_runtime_table->hostLayerState(layer_idx);
                        const std::uint64_t required_route_slots =
                            static_cast<std::uint64_t>(total_tokens) *
                            static_cast<std::uint64_t>(config_.moe.top_k);
                        if (!runtime_layer.route_participant_ids ||
                            required_route_slots == 0u ||
                            required_route_slots >
                                static_cast<std::uint64_t>(
                                    runtime_layer.prefill_route_capacity))
                        {
                            throw std::runtime_error(
                                "Qwen35 MoE captured overlay continuation has no complete final device route assignment ledger");
                        }
                        pinned_domain_route_assignment = {
                            .participant_ids =
                                runtime_layer.route_participant_ids,
                            .capacity =
                                runtime_layer.prefill_route_capacity,
                        };
                        pinned_runtime_route_weights =
                            bindMoERuntimeRouteWeights(
                                moe_runtime_table->deviceLayerState(
                                    layer_idx),
                                runtime_layer,
                                captured_overlay_route_weight_projection);
                        if (!pinned_runtime_route_weights.validFor(
                                static_cast<std::uint32_t>(total_tokens),
                                static_cast<std::uint32_t>(
                                    config_.moe.top_k)))
                        {
                            throw std::runtime_error(
                                "Qwen35 MoE captured overlay continuation has no workload-correct final device route-weight publication");
                        }
                    }
                    else if (mapped_activation_topology &&
                             !sparse_graph_contract
                                  .ownsDispatchAuthority())
                    {
                        /*
                         * The logical root captures one packet publisher per
                         * remote lane before both continuation GPUs enter the
                         * common local-expert wave. A sibling with no packet
                         * work must still join those capture lifecycles in the
                         * same order; passive waves express that absence
                         * without launching an empty graph or routing traffic
                         * through the host.
                         */
                        graph.setGraphCaptureWaveContract(
                            prefix + "moe_routing",
                            GraphCaptureWaveContract{
                                .identity = captured_overlay_ticket_wave,
                                .passive_following_identities =
                                    mapped_activation_dispatch_wave_identities,
                            });
                    }
                }

                if (sparse_graph_contract.ownsDispatchAuthority() &&
                    (!use_mapped_activation_parent ||
                     !mapped_activation_local_participants.empty()))
                {
                    ensure_host_dispatch_path();
                }

                /**
                 * Exact endpoint whose ordered return closes host dispatch.
                 *
                 * Rank-batched remote groups are lowered before rank-local
                 * endpoints. Keeping this as one typed endpoint prevents a
                 * remote batch and a later loopback return from both claiming
                 * the same residency lease terminal.
                 */
                struct OrderedHostSparseReturnEndpoint
                {
                    size_t tier_index = 0;
                    int participant_id = -1;

                    [[nodiscard]] bool matches(
                        size_t tier,
                        int participant) const noexcept
                    {
                        return tier_index == tier &&
                               participant_id == participant;
                    }
                };
                std::optional<OrderedHostSparseReturnEndpoint>
                    final_overlay_participant;
                std::optional<OrderedHostSparseReturnEndpoint>
                    final_rank_local_overlay_participant;
                for (size_t tier_index = 0;
                     tier_index < overlay_plan->routed_tiers.size();
                     ++tier_index)
                {
                    const auto tier_mask = expertMaskForTier(
                        *overlay_placement,
                        config_.moe.num_experts,
                        static_cast<int>(tier_index));
                    if (!hasActiveExpertMask(tier_mask))
                        continue;
                    const auto targets =
                        resolve_overlay_tier_targets(tier_index);
                    for (const int target_participant :
                         targets.participants)
                    {
                        const auto participant_mask =
                            owner_map_lifetime->expertMaskForParticipant(
                                layer_idx,
                                target_participant,
                                config_.moe.num_experts);
                        if (hasActiveExpertMask(participant_mask))
                        {
                            final_overlay_participant =
                                OrderedHostSparseReturnEndpoint{
                                    .tier_index = tier_index,
                                    .participant_id = target_participant,
                                };
                        }
                        const auto *const participant_descriptor =
                            owner_map_lifetime->participantForId(
                                target_participant);
                        if (distributed_overlay &&
                            sparse_graph_contract
                                .ownsDispatchAuthority() &&
                            participant_descriptor &&
                            participant_descriptor->world_rank_known &&
                            participant_descriptor->world_rank ==
                                config_.moe.overlay_mpi_ctx->rank() &&
                            (hasActiveExpertMask(participant_mask) ||
                             use_mapped_activation_parent))
                        {
                            /* The rank-local ticket contains only colocated
                             * lower-tier returns, so its completion frontier
                             * is independent of mapped remote lanes. */
                            final_rank_local_overlay_participant =
                                OrderedHostSparseReturnEndpoint{
                                    .tier_index = tier_index,
                                    .participant_id = target_participant,
                                };
                        }
                    }
                }
                const std::optional<OrderedHostSparseReturnEndpoint>
                    final_host_sparse_return =
                        final_rank_local_overlay_participant
                            ? final_rank_local_overlay_participant
                            : final_overlay_participant;

                std::vector<std::shared_ptr<MoEOverlayCollectiveWorkspace>> participant_workspaces(
                    static_cast<size_t>(participant_count));
                for (int participant = 0;
                     participant < participant_count;
                     ++participant)
                {
                    participant_workspaces[static_cast<size_t>(participant)] =
                        overlayProtocolWorkspaceForParticipant(
                            device, participant);
                }

                std::shared_ptr<IMoEOverlaySparseCollectiveContext>
                    collective_context_lifetime;
                std::shared_ptr<IMoEOverlaySparseCollectiveContext>
                    rank_local_collective_context_lifetime;
                if (!distributed_overlay)
                {
                    MoEOverlayLocalSparseCollectiveContext::Config
                        collective_config;
                    collective_config.participant_count =
                        participant_count;
                    collective_config.slot_count = std::max<size_t>(
                        8,
                        overlay_plan->routed_tiers.size() *
                                static_cast<size_t>(participant_count) * 4u +
                            8u);
                    collective_context_lifetime =
                        std::make_shared<
                            MoEOverlayLocalSparseCollectiveContext>(
                            collective_config);
                }

                const auto ensure_rank_local_collective_context = [&]()
                {
                    if (rank_local_collective_context_lifetime)
                        return;
                    MoEOverlayRankLocalSparseCollectiveContext::Config
                        rank_local_config;
                    rank_local_config.slot_count = std::max<size_t>(
                        8,
                        overlay_plan->routed_tiers.size() *
                                static_cast<size_t>(participant_count) * 4u +
                            8u);
                    rank_local_collective_context_lifetime =
                        std::make_shared<
                            MoEOverlayRankLocalSparseCollectiveContext>(
                                rank_local_config);
                };

                if (captured_overlay_continuation && !distributed_overlay)
                {
                    /* One continuation GPU and colocated lower tiers form
                     * direct point-to-point sparse edges. They must not mimic
                     * a participant-count collective merely because MPI has a
                     * single rank. */
                    ensure_rank_local_collective_context();
                    collective_context_lifetime =
                        rank_local_collective_context_lifetime;
                }

                std::string last_return_reduce;
                bool first_return_scatter = true;
                std::vector<std::string>
                    distributed_rank_batch_dispatch_nodes;
                std::vector<std::string>
                    mapped_activation_dispatch_nodes;
                struct MappedActivationReturnBinding
                {
                    std::string dispatch_node_name;
                    std::string node_name;
                    MoEOverlayMappedActivationDeviceLane lane;
                    std::uint32_t stage_ordinal = 0u;
                };
                std::vector<MappedActivationReturnBinding>
                    mapped_activation_return_bindings;
                std::vector<std::string>
                    mapped_activation_return_nodes;
                bool has_rank_local_canonical_ticket_returns = false;
                std::string local_canonical_ticket_terminal;
                struct PendingCanonicalTicketConsume
                {
                    std::string participant_suffix;
                    std::shared_ptr<
                        MoEOverlayCanonicalRouteReturnTicketStorage>
                        storage;
                };
                std::vector<PendingCanonicalTicketConsume>
                    pending_canonical_ticket_consumes;

                const auto routed_domain_ordinal_for_tier =
                    [&](const RoutedExpertTier &tier) -> int
                {
                    for (size_t candidate = 0;
                         candidate < overlay_plan->domains.size();
                         ++candidate)
                    {
                        if (overlay_plan->domains[candidate].name ==
                            tier.domain)
                        {
                            return static_cast<int>(candidate);
                        }
                    }
                    throw std::logic_error(
                        "Qwen35 MoE rank-batch tier has no stable routed-domain ordinal");
                };

                /**
                 * Build one CPU expert endpoint inside a full NodeTP follower
                 * graph. The caller owns the rank-batch receive/send stages;
                 * this helper owns only prepared expert state and the exact
                 * dispatch-to-compute edge. Keeping that boundary explicit
                 * prevents the follower from acquiring route or publication
                 * authority merely because it also carries dense TP layers.
                 */
                const auto add_rank_batch_target_local_expert =
                    [&](size_t tier_index,
                        int target_participant,
                        const std::shared_ptr<MoEOverlaySparseRows> &inbound,
                        const std::shared_ptr<MoEOverlayReturnRows> &outbound,
                        const std::string &dispatch_node) -> std::string
                {
                    const auto &tier =
                        overlay_plan->routed_tiers[tier_index];
                    const auto *participant =
                        owner_map_lifetime->participantForId(
                            target_participant);
                    if (!participant || !participant->world_rank_known ||
                        participant->world_rank !=
                            config_.moe.overlay_mpi_ctx->rank())
                    {
                        throw std::logic_error(
                            "Qwen35 MoE rank-batch target does not own its declared participant");
                    }

                    const DeviceId target_device =
                        participantDeviceForGraphNativeOverlay(
                            *owner_map_lifetime,
                            target_participant);
                    if (!device.is_cpu() || !target_device.is_cpu())
                    {
                        throw std::logic_error(
                            "Qwen35 MoE full-model rank-batch follower is the CPU NodeTP protocol; captured GPU peers require the retained device transaction role");
                    }

                    auto participant_mask =
                        owner_map_lifetime->expertMaskForParticipant(
                            layer_idx,
                            target_participant,
                            config_.moe.num_experts);
                    if (!hasActiveExpertMask(participant_mask))
                    {
                        throw std::logic_error(
                            "Qwen35 MoE rank-batch follower was asked to build an inactive participant endpoint");
                    }

                    MoELocalExpertStage::Params local_params;
                    local_params.device_id = target_device;
                    local_params.input_rows_lifetime = inbound;
                    local_params.output_rows_lifetime = outbound;
                    local_params.workspace_lifetime =
                        participant_workspaces[static_cast<size_t>(
                            target_participant)];
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
                    local_params.runtime_participant_index =
                        target_participant;
                    const size_t compact_row_capacity =
                        static_cast<size_t>(
                            overlaySparseGraphRowCapacity(
                                config_,
                                target_device,
                                total_tokens));
                    local_params.graph_row_capacity = compact_row_capacity;
                    local_params.serial_compact_buffer_arena =
                        localExpertSerialBufferArenaForParticipant(
                            target_device,
                            target_participant,
                            std::max<size_t>(compact_row_capacity, 1u));
                    local_params.expert_weight_resolution_policy =
                        MoELocalExpertStage::ExpertWeightResolutionPolicy::
                            RegistryOnly;

                    if (model_ctx_)
                    {
                        const auto weight_manager =
                            model_ctx_->concreteWeightManager();
                        if (weight_manager)
                        {
                            auto &registry =
                                weight_manager->expertGemmRegistry();
                            local_params.expert_registry = &registry;
                            (void)registry
                                .populateExpertEnginesForParticipant(
                                    tier.domain,
                                    target_device,
                                    participant->world_rank,
                                    participant
                                        ->domain_participant_index,
                                    layer_idx,
                                    config_.moe.num_experts,
                                    local_params.prepared_gate_gemm,
                                    local_params.prepared_up_gemm,
                                    local_params.prepared_down_gemm);
                        }
                    }
                    if (!local_params.expert_registry ||
                        !MoELocalExpertStage::prepareExpertGemmEngines(
                            local_params))
                    {
                        throw std::runtime_error(
                            "Qwen35 MoE rank-batch follower has incomplete prepared expert engines for layer " +
                            std::to_string(layer_idx) + " participant " +
                            std::to_string(target_participant));
                    }

                    if (config_.moe.expert_overlay_participant_residency)
                    {
                        auto endpoint = config_.moe
                                            .expert_overlay_participant_residency
                                            ->endpoint(target_participant);
                        if (!endpoint)
                        {
                            throw std::runtime_error(
                                "Qwen35 MoE rank-batch follower has no prepared residency endpoint for participant " +
                                std::to_string(target_participant));
                        }
                        std::vector<MoEOverlayPreparedExpertTriplet>
                            prepared_triplets;
                        std::string residency_error;
                        if (!resolveMoEOverlayPreparedExpertTriplets(
                                *local_params.expert_registry,
                                *participant,
                                layer_idx,
                                config_.moe.num_experts,
                                local_params.expert_mask,
                                prepared_triplets,
                                &residency_error) ||
                            !config_.moe
                                 .expert_overlay_participant_residency
                                 ->registerInitialLayer(
                                     target_participant,
                                     layer_idx,
                                     local_params.expert_mask,
                                     prepared_triplets,
                                     &residency_error))
                        {
                            throw std::runtime_error(
                                "Qwen35 MoE rank-batch follower could not publish prepared residency for layer " +
                                std::to_string(layer_idx) + " participant " +
                                std::to_string(target_participant) + ": " +
                                residency_error);
                        }
                        local_params.overlay_participant_residency =
                            std::move(endpoint);
                    }

                    const std::string local_name =
                        prefix + "moe_rank_batch_target_" +
                        nodeSuffixForTier(
                            tier,
                            static_cast<int>(tier_index)) +
                        "_p" + std::to_string(target_participant) +
                        "_local_expert";
                    graph.addNode(
                        local_name,
                        ComputeStageFactory::createMoELocalExpert(
                            local_params),
                        target_device);
                    graph.addDependency(local_name, dispatch_node);
                    return local_name;
                };

                for (size_t tier_index = 0; tier_index < overlay_plan->routed_tiers.size(); ++tier_index)
                {
                    const auto &tier = overlay_plan->routed_tiers[tier_index];
                    auto tier_mask = expertMaskForTier(*overlay_placement,
                                                       config_.moe.num_experts,
                                                       static_cast<int>(tier_index));
                    if (!hasActiveExpertMask(tier_mask) &&
                        !use_mapped_activation_parent)
                    {
                        LOG_DEBUG("[Qwen35MoEGraph] Skipping inactive MoE expert overlay tier "
                                  << tier.name << " for layer " << layer_idx);
                        continue;
                    }

                    const auto tier_targets =
                        resolve_overlay_tier_targets(tier_index);
                    std::vector<int> target_participants =
                        tier_targets.participants;
                    const bool compute_apportioned_tier_on_graph_local_participant =
                        tier_targets.graph_local_apportioned;

                    if (distributed_overlay && !target_participants.empty())
                    {
                        const auto *continuation_root_descriptor =
                            owner_map_lifetime->participantForId(
                                continuation_root_participant);
                        if (!continuation_root_descriptor ||
                            !continuation_root_descriptor->world_rank_known)
                        {
                            throw std::logic_error(
                                "Qwen35 MoE rank-batch graph cannot resolve its authenticated continuation participant");
                        }
                        const int source_world_rank =
                            continuation_root_descriptor->world_rank;
                        const int domain_ordinal =
                            routed_domain_ordinal_for_tier(tier);

                        if (sparse_graph_contract.isRankBatchTarget())
                        {
                            if (cpu_llep_state)
                            {
                                throw std::logic_error(
                                    "Qwen35 MoE full-model rank-batch followers cannot enter the unfinished rank-local CPU LLEP child protocol");
                            }
                            const int target_world_rank =
                                config_.moe.overlay_mpi_ctx->rank();
                            std::vector<int> local_participants;
                            std::sort(
                                target_participants.begin(),
                                target_participants.end());
                            for (const int participant_id :
                                 target_participants)
                            {
                                const auto *participant =
                                    owner_map_lifetime->participantForId(
                                        participant_id);
                                const auto participant_mask =
                                    owner_map_lifetime
                                        ->expertMaskForParticipant(
                                            layer_idx,
                                            participant_id,
                                            config_.moe.num_experts);
                                if (participant &&
                                    participant->world_rank_known &&
                                    participant->world_rank ==
                                        target_world_rank &&
                                    hasActiveExpertMask(participant_mask))
                                {
                                    local_participants.push_back(
                                        participant_id);
                                }
                            }
                            if (local_participants.empty())
                                continue;

                            auto transport =
                                overlayRankBatchTransportForGroup(
                                    device,
                                    static_cast<int>(tier_index),
                                    domain_ordinal,
                                    source_world_rank,
                                    target_world_rank,
                                    local_participants,
                                    *owner_map_lifetime,
                                    continuation_root_participant);
                            const auto make_rank_batch_key =
                                [&](MoEOverlayCollectiveDirection direction)
                            {
                                if (mtp_sidecar_context)
                                {
                                    return makeMTPMoEOverlayRankBatchKey(
                                        1,
                                        0,
                                        mtp_depth_idx,
                                        layer_idx,
                                        static_cast<int>(tier_index),
                                        domain_ordinal,
                                        source_world_rank,
                                        target_world_rank,
                                        direction);
                                }
                                return makeMoEOverlayRankBatchKey(
                                    1,
                                    0,
                                    ExpertHistogramSource::DecodeToken,
                                    layer_idx,
                                    static_cast<int>(tier_index),
                                    domain_ordinal,
                                    source_world_rank,
                                    target_world_rank,
                                    direction);
                            };

                            std::vector<std::shared_ptr<
                                MoEOverlaySparseRows>> dispatch_inbound;
                            std::vector<std::shared_ptr<
                                MoEOverlayReturnRows>> local_outputs;
                            dispatch_inbound.reserve(
                                local_participants.size());
                            local_outputs.reserve(
                                local_participants.size());
                            for (const int participant_id :
                                 local_participants)
                            {
                                const auto &workspace =
                                    participant_workspaces
                                        [static_cast<size_t>(
                                            participant_id)];
                                dispatch_inbound.push_back(
                                    std::make_shared<
                                        MoEOverlaySparseRows>(
                                        transport->hasSharedRowStorage()
                                            ? transport
                                                  ->sharedDispatchRows(
                                                      participant_id)
                                            : workspace->dispatchReceive(
                                                  layer_idx,
                                                  static_cast<int>(
                                                      tier_index))));
                                local_outputs.push_back(
                                    std::make_shared<
                                        MoEOverlayReturnRows>(
                                        transport->hasSharedRowStorage()
                                            ? transport
                                                  ->sharedReturnRows(
                                                      participant_id)
                                            : workspace->localExpertOutput(
                                                  layer_idx,
                                                  static_cast<int>(
                                                      tier_index))));
                            }

                            MoERankBatchDispatchStage::Params
                                follower_dispatch;
                            follower_dispatch.device_id = DeviceId::cpu();
                            follower_dispatch.endpoint_role =
                                MoERankBatchEndpointRole::RemoteTarget;
                            follower_dispatch.transport = transport;
                            follower_dispatch.key = make_rank_batch_key(
                                MoEOverlayCollectiveDirection::Dispatch);
                            follower_dispatch.source_participant =
                                continuation_root_participant;
                            follower_dispatch.participant_ids =
                                local_participants;
                            follower_dispatch.inbound_rows =
                                dispatch_inbound;
                            follower_dispatch.seq_len = total_tokens;
                            follower_dispatch.top_k = config_.moe.top_k;
                            follower_dispatch.d_model = config_.d_model;
                            follower_dispatch.tier_index =
                                static_cast<int>(tier_index);

                            const std::string group_suffix =
                                nodeSuffixForTier(
                                    tier,
                                    static_cast<int>(tier_index)) +
                                "_rank" +
                                std::to_string(target_world_rank);
                            const std::string dispatch_name =
                                prefix +
                                "moe_rank_batch_target_dispatch_" +
                                group_suffix;
                            graph.addNode(
                                dispatch_name,
                                ComputeStageFactory::
                                    createMoERankBatchDispatch(
                                        follower_dispatch),
                                DeviceId::cpu());
                            graph.addDependency(
                                dispatch_name,
                                last_return_reduce.empty()
                                    ? dispatch_dependency
                                    : last_return_reduce);

                            std::vector<std::string> local_nodes;
                            local_nodes.reserve(
                                local_participants.size());
                            for (size_t participant_index = 0;
                                 participant_index <
                                 local_participants.size();
                                 ++participant_index)
                            {
                                local_nodes.push_back(
                                    add_rank_batch_target_local_expert(
                                        tier_index,
                                        local_participants[
                                            participant_index],
                                        dispatch_inbound[
                                            participant_index],
                                        local_outputs[
                                            participant_index],
                                        dispatch_name));
                            }

                            std::vector<std::shared_ptr<const
                                MoEOverlayReturnRows>> outbound_rows;
                            outbound_rows.reserve(local_outputs.size());
                            for (const auto &rows : local_outputs)
                                outbound_rows.push_back(rows);

                            MoERankBatchReturnReduceStage::Params
                                follower_return;
                            follower_return.device_id = DeviceId::cpu();
                            follower_return.endpoint_role =
                                MoERankBatchEndpointRole::RemoteTarget;
                            follower_return.transport = transport;
                            follower_return.key = make_rank_batch_key(
                                MoEOverlayCollectiveDirection::ReturnReduce);
                            follower_return.continuation_participant =
                                continuation_root_participant;
                            follower_return.participant_ids =
                                local_participants;
                            follower_return.outbound_rows =
                                std::move(outbound_rows);
                            follower_return.seq_len = total_tokens;
                            follower_return.d_model = config_.d_model;

                            const std::string return_name =
                                prefix +
                                "moe_rank_batch_target_return_" +
                                group_suffix;
                            graph.addNode(
                                return_name,
                                ComputeStageFactory::
                                    createMoERankBatchReturnReduce(
                                        follower_return),
                                DeviceId::cpu());
                            for (const auto &local_node : local_nodes)
                            {
                                graph.addDependency(
                                    return_name,
                                    local_node);
                            }
                            last_return_reduce = return_name;
                            continue;
                        }

                        if (!sparse_graph_contract
                                 .ownsDispatchAuthority() ||
                            source_world_rank !=
                                config_.moe.overlay_mpi_ctx->rank())
                        {
                            throw std::logic_error(
                                "Qwen35 MoE rank-batch source role does not own the authenticated continuation participant");
                        }
                        if (cpu_llep_state)
                        {
                            throw std::logic_error(
                                "Qwen35 MoE rank-batch source cannot be combined with the unfinished rank-local CPU LLEP child protocol");
                        }

                        /*
                         * One remote rank may own several logical GPU/CPU
                         * participants. Keep those participant packets
                         * separate, but move them in one authenticated message.
                         * Portable rank batches omit empty initial masks
                         * symmetrically. Mapped retained epochs bind every
                         * topology-declared lane so later migration can make an
                         * initially empty participant live without recapture.
                         */
                        std::map<int, std::vector<int>> participants_by_rank;
                        std::vector<int> canonical_remote_participants;
                        std::vector<int> local_rank_participants;
                        std::sort(
                            target_participants.begin(),
                            target_participants.end());
                        for (const int target_participant : target_participants)
                        {
                            const auto participant_mask =
                                owner_map_lifetime->expertMaskForParticipant(
                                    layer_idx,
                                    target_participant,
                                    config_.moe.num_experts);
                            if (!use_mapped_activation_parent &&
                                !hasActiveExpertMask(participant_mask))
                                continue;
                            const auto *participant =
                                owner_map_lifetime->participantForId(
                                    target_participant);
                            if (!participant ||
                                !participant->world_rank_known)
                            {
                                throw std::logic_error(
                                    "Qwen35 MoE distributed sparse tier contains an unresolved participant endpoint");
                            }
                            if (participant->world_rank == source_world_rank)
                            {
                                /* One rank may own both continuation and a
                                 * NodeTP lower-tier endpoint. Preserve that
                                 * endpoint as an explicit in-process sparse
                                 * edge rather than inventing a self-MPI lane. */
                                local_rank_participants.push_back(
                                    target_participant);
                                continue;
                            }
                            participants_by_rank[participant->world_rank]
                                .push_back(target_participant);
                            canonical_remote_participants.push_back(
                                target_participant);
                        }

                        std::vector<std::pair<int, std::vector<int>>> rank_groups;
                        rank_groups.reserve(participants_by_rank.size());
                        for (auto &[target_rank, participants] :
                             participants_by_rank)
                        {
                            std::sort(participants.begin(), participants.end());
                            rank_groups.emplace_back(
                                target_rank, std::move(participants));
                        }
                        std::sort(
                            rank_groups.begin(),
                            rank_groups.end(),
                            [](const auto &lhs, const auto &rhs)
                            {
                                return lhs.second.front() < rhs.second.front();
                            });
                        std::vector<int> batched_participant_order;
                        for (const auto &group : rank_groups)
                        {
                            batched_participant_order.insert(
                                batched_participant_order.end(),
                                group.second.begin(),
                                group.second.end());
                        }
                        if (batched_participant_order !=
                            canonical_remote_participants)
                        {
                            throw std::logic_error(
                                "Qwen35 MoE cannot rank-batch an interleaved participant order without changing canonical FP32 accumulation order");
                        }

                        for (const auto &[target_world_rank, participants] :
                             rank_groups)
                        {
                            auto transport =
                                overlayRankBatchTransportForGroup(
                                    device,
                                    static_cast<int>(tier_index),
                                    domain_ordinal,
                                    source_world_rank,
                                    target_world_rank,
                                    participants,
                                    *owner_map_lifetime,
                                    continuation_root_participant);
                            const std::string group_suffix =
                                nodeSuffixForTier(
                                    tier,
                                    static_cast<int>(tier_index)) +
                                "_rank" +
                                std::to_string(target_world_rank);

                            if (use_mapped_activation_parent)
                            {
                                auto *const mapped_transport = dynamic_cast<
                                    IMoEOverlayMappedActivationTransport *>(
                                    transport.get());
                                if (!mapped_transport ||
                                    transport->kind() !=
                                        MoEOverlayRankBatchTransportKind::
                                            NodeLocalSharedRows ||
                                    mapped_activation_graph_family_ordinal >=
                                        mapped_transport
                                            ->activationGraphFamilyCount())
                                {
                                    throw std::runtime_error(
                                        "Qwen35 MoE node-local activation plan did not produce its required mapped graph capability");
                                }

                                const std::uint32_t stage_ordinal =
                                    mapped_transport->activationStageOrdinal(
                                        mapped_activation_graph_family_ordinal,
                                        layer_idx);
                                for (const int participant : participants)
                                {
                                    auto lane =
                                        mapped_transport->activationDeviceLane(
                                            participant,
                                            mapped_activation_graph_family_ordinal,
                                            device);
                                    if (!lane.valid() ||
                                        lane.target_participant_id !=
                                            participant ||
                                        lane.graph_family_ordinal !=
                                            mapped_activation_graph_family_ordinal)
                                    {
                                        throw std::runtime_error(
                                            "Qwen35 MoE mapped activation lane diverged from the planner participant or graph family");
                                    }

                                    const std::string packet_name =
                                        prefix +
                                        "moe_overlay_activation_dispatch_pack_" +
                                        group_suffix + "_p" +
                                        std::to_string(participant);
                                    mapped_activation_return_bindings.push_back(
                                        MappedActivationReturnBinding{
                                            .dispatch_node_name = packet_name,
                                            .node_name =
                                                prefix +
                                                "moe_overlay_activation_return_consume_" +
                                                group_suffix + "_p" +
                                                std::to_string(participant),
                                            .lane = std::move(lane),
                                            .stage_ordinal = stage_ordinal,
                                        });
                                }
                                continue;
                            }

                            ensure_host_dispatch_path();
                            std::vector<std::shared_ptr<
                                MoEOverlayCollectiveWorkspace>>
                                group_workspaces;
                            group_workspaces.reserve(participants.size());
                            for (const int participant : participants)
                            {
                                group_workspaces.push_back(
                                    participant_workspaces
                                        [static_cast<size_t>(participant)]);
                            }

                            const auto make_rank_batch_key =
                                [&](MoEOverlayCollectiveDirection direction)
                            {
                                if (mtp_sidecar_context)
                                {
                                    return makeMTPMoEOverlayRankBatchKey(
                                        1,
                                        0,
                                        mtp_depth_idx,
                                        layer_idx,
                                        static_cast<int>(tier_index),
                                        domain_ordinal,
                                        source_world_rank,
                                        target_world_rank,
                                        direction);
                                }
                                return makeMoEOverlayRankBatchKey(
                                    1,
                                    0,
                                    ExpertHistogramSource::DecodeToken,
                                    layer_idx,
                                    static_cast<int>(tier_index),
                                    domain_ordinal,
                                    source_world_rank,
                                    target_world_rank,
                                    direction);
                            };

                            MoERankBatchDispatchStage::Params batch_dispatch;
                            batch_dispatch.device_id = DeviceId::cpu();
                            batch_dispatch.endpoint_role =
                                MoERankBatchEndpointRole::ContinuationSource;
                            batch_dispatch.transport = transport;
                            batch_dispatch.key = make_rank_batch_key(
                                MoEOverlayCollectiveDirection::Dispatch);
                            batch_dispatch.source_participant =
                                continuation_root_participant;
                            batch_dispatch.participant_ids = participants;
                            if (!transport->hasSharedRowStorage())
                            {
                                batch_dispatch.participant_workspaces =
                                    group_workspaces;
                            }
                            batch_dispatch.dispatch_output =
                                dispatch_output_lifetime;
                            batch_dispatch.ticket_storage =
                                dispatch_ticket_storage;
                            if (!dispatch_ticket_storage)
                            {
                                batch_dispatch.hidden = buffers.normalized;
                                batch_dispatch.routing_indices =
                                    routing_indices;
                                batch_dispatch.routing_weights =
                                    routing_weights;
                                batch_dispatch.hidden_buffer_id =
                                    buffers.idFor(BufferId::NORMALIZED);
                                batch_dispatch.routing_indices_buffer_id =
                                    buffers.idFor(
                                        BufferId::MOE_EXPERT_INDICES);
                                batch_dispatch.routing_weights_buffer_id =
                                    buffers.idFor(
                                        BufferId::MOE_EXPERT_WEIGHTS);
                            }
                            batch_dispatch.seq_len = total_tokens;
                            batch_dispatch.top_k = config_.moe.top_k;
                            batch_dispatch.d_model = config_.d_model;
                            batch_dispatch.tier_index =
                                static_cast<int>(tier_index);

                            const std::string batch_dispatch_name =
                                prefix + "moe_rank_batch_dispatch_" +
                                group_suffix;
                            graph.addNode(
                                batch_dispatch_name,
                                ComputeStageFactory::
                                    createMoERankBatchDispatch(
                                        batch_dispatch),
                                DeviceId::cpu());
                            graph.addDependency(
                                batch_dispatch_name,
                                dispatch_dependency);
                            distributed_rank_batch_dispatch_nodes.push_back(
                                batch_dispatch_name);
                            if (!last_return_reduce.empty())
                            {
                                graph.addDependency(
                                    batch_dispatch_name,
                                    last_return_reduce);
                            }

                            std::vector<std::shared_ptr<
                                MoEOverlayReturnRows>>
                                return_inbound;
                            return_inbound.reserve(participants.size());
                            for (size_t participant_index = 0;
                                 participant_index < participants.size();
                                 ++participant_index)
                            {
                                const int participant =
                                    participants[participant_index];
                                return_inbound.push_back(
                                    std::make_shared<MoEOverlayReturnRows>(
                                        transport->hasSharedRowStorage()
                                            ? transport->sharedReturnRows(
                                                  participant)
                                            : group_workspaces
                                                  [participant_index]
                                                      ->returnReceive(
                                                          layer_idx,
                                                          static_cast<int>(
                                                              tier_index))));
                            }

                            const bool is_final_ordered_return =
                                final_host_sparse_return.has_value() &&
                                final_host_sparse_return->tier_index ==
                                    tier_index &&
                                std::find(
                                    participants.begin(),
                                    participants.end(),
                                    final_host_sparse_return
                                        ->participant_id) !=
                                    participants.end();
                            MoERankBatchReturnReduceStage::Params batch_return;
                            batch_return.device_id = DeviceId::cpu();
                            batch_return.endpoint_role =
                                MoERankBatchEndpointRole::ContinuationSource;
                            batch_return.transport = transport;
                            batch_return.key = make_rank_batch_key(
                                MoEOverlayCollectiveDirection::ReturnReduce);
                            batch_return.continuation_participant =
                                continuation_root_participant;
                            batch_return.participant_ids = participants;
                            batch_return.inbound_rows =
                                std::move(return_inbound);
                            batch_return.ticket_storage =
                                dispatch_ticket_storage;
                            if (!dispatch_ticket_storage)
                            {
                                batch_return.dense_output = moe_output;
                                batch_return.dense_output_buffer_id =
                                    buffers.idFor(
                                        BufferId::MOE_COMBINED_OUTPUT);
                            }
                            batch_return.seq_len = total_tokens;
                            batch_return.d_model = config_.d_model;
                            batch_return.clear_output_before_scatter =
                                first_return_scatter;
                            batch_return.publish_ticket_completion =
                                dispatch_ticket_storage &&
                                is_final_ordered_return;
                            batch_return.dispatch_output =
                                dispatch_output_lifetime;
                            batch_return.residency_lease_terminal =
                                moeOverlayHostDispatchLeaseTerminal(
                                    host_dispatch_lease_owner,
                                    is_final_ordered_return);

                            const std::string batch_return_name =
                                prefix + "moe_rank_batch_return_reduce_" +
                                group_suffix;
                            graph.addNode(
                                batch_return_name,
                                ComputeStageFactory::
                                    createMoERankBatchReturnReduce(
                                        batch_return),
                                DeviceId::cpu());
                            graph.addDependency(
                                batch_return_name,
                                batch_dispatch_name);

                            std::string tier_terminal = batch_return_name;
                            if (dispatch_ticket_storage &&
                                is_final_ordered_return)
                            {
                                MoEOverlayTicketConsumeStage::Params
                                    consume_params;
                                consume_params.device_id = device;
                                consume_params.output =
                                    captured_overlay_continuation
                                        ? buffers.attn_proj
                                        : moe_output;
                                consume_params.output_buffer_id =
                                    captured_overlay_continuation
                                        ? buffers.idFor(BufferId::ATTN_PROJ)
                                        : buffers.idFor(
                                              BufferId::MOE_COMBINED_OUTPUT);
                                consume_params.layer_idx = layer_idx;
                                consume_params.bucket_rows = total_tokens;
                                consume_params.d_model = config_.d_model;
                                consume_params.ticket_storage =
                                    dispatch_ticket_storage;

                                const std::string consume_name =
                                    prefix +
                                    "moe_overlay_ticket_consume_" +
                                    group_suffix;
                                graph.addNode(
                                    consume_name,
                                    ComputeStageFactory::
                                        createMoEOverlayTicketConsume(
                                            consume_params),
                                    device);
                                graph.addDependency(
                                    consume_name,
                                    batch_return_name);
                                if (captured_overlay_continuation)
                                {
                                    graph.setGraphCaptureWaveContract(
                                        consume_name,
                                        GraphCaptureWaveContract{
                                            .identity =
                                                captured_overlay_inbound_wave,
                                        });
                                }
                                tier_terminal = consume_name;
                            }

                            last_return_reduce = tier_terminal;
                            first_return_scatter = false;
                        }
                        if (local_rank_participants.empty())
                            continue;

                        /*
                         * Remote rank batches and mapped lanes have now been
                         * declared. Fall through with only colocated endpoints
                         * and bind them to the allocation-free rank-local
                         * context. This includes the continuation's own
                         * participant loopback and colocated lower-tier peers.
                         * Their host dispatch branch shares the immutable
                         * ticket publication fork with continuation GPU work.
                         */
                        target_participants =
                            std::move(local_rank_participants);
                        ensure_host_dispatch_path();
                        ensure_rank_local_collective_context();
                        collective_context_lifetime =
                            rank_local_collective_context_lifetime;
                    }

                    for (const int target_participant : target_participants)
                    {
                        auto participant_mask =
                            owner_map_lifetime->expertMaskForParticipant(
                                layer_idx,
                                target_participant,
                                config_.moe.num_experts);
                        if (!hasActiveExpertMask(participant_mask) &&
                            !(distributed_overlay &&
                              use_mapped_activation_parent))
                            continue;

                        const DeviceId target_device = participantDeviceForGraphNativeOverlay(
                            *owner_map_lifetime,
                            target_participant);
                        const bool direct_rank_local_protocol =
                            rank_local_collective_context_lifetime &&
                            collective_context_lifetime.get() ==
                                rank_local_collective_context_lifetime.get();
                        if (!distributed_overlay &&
                            target_device.is_gpu() &&
                            target_device != device)
                        {
                            throw std::runtime_error(
                                "Qwen35 MoE graph-native overlay would execute GPU expert participant " +
                                std::to_string(target_participant) + " on " +
                                target_device.to_string() + " from graph device " +
                                device.to_string() +
                                ". LocalTP expert-ID-apportioned GPU domains must lower only the graph-local participant; "
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
                        const std::vector<int> dispatch_sources =
                            distributed_overlay || direct_rank_local_protocol
                                ? std::vector<int>{
                                      continuation_root_participant}
                                : participantsWithLast(
                                      participant_count,
                                      target_participant);
                        for (const int source_participant : dispatch_sources)
                        {
                            /*
                             * In a distributed overlay every rank enters the
                             * same sparse collective, but only the rank that
                             * owns target_participant receives its packet.
                             * The receiving local-expert stage always reads
                             * target_dispatch_inbound, so the collective must
                             * publish into that target-owned view on every
                             * rank.  Publishing into the source workspace
                             * would leave remote owners with an empty input
                             * even though MPI delivered their routes.
                             *
                             * The single-process protocol retains its
                             * source-scoped scratch views because it models
                             * every logical participant in one graph.
                             */
                            auto inbound_lifetime =
                                distributed_overlay ||
                                        direct_rank_local_protocol ||
                                        source_participant == target_participant
                                    ? target_dispatch_inbound
                                    : std::make_shared<MoEOverlaySparseRows>(
                                          participant_workspaces
                                              [static_cast<size_t>(
                                                  source_participant)]
                                                  ->dispatchReceive(
                                                      layer_idx,
                                                      static_cast<int>(
                                                          tier_index)));

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
                            const bool publishes_dispatch_payload =
                                sparse_graph_contract
                                    .ownsDispatchAuthority() &&
                                source_participant ==
                                    continuation_root_participant;
                            sparse_dispatch_params.payload_publication_role =
                                publishes_dispatch_payload
                                    ? MoESparseDispatchStage::
                                          PayloadPublicationRole::
                                              PayloadAuthority
                                    : MoESparseDispatchStage::
                                          PayloadPublicationRole::
                                              EmptyCollectiveParticipant;
                            sparse_dispatch_params.replicated_hidden_export = true;
                            sparse_dispatch_params.logical_continuation_root_participant = continuation_root_participant;
                            sparse_dispatch_params.manual_boundary_requires_collective_completion =
                                distributed_overlay ||
                                direct_rank_local_protocol ||
                                source_participant == target_participant;
                            /*
                             * Independent rank graphs do not share stage-object
                             * lifetime. Require the runner's request/chunk
                             * identity instead of deriving a sparse MPI key
                             * from a local execution counter.
                             */
                            sparse_dispatch_params.require_explicit_transaction_identity =
                                distributed_overlay;
                            sparse_dispatch_params.require_explicit_execution_semantics =
                                config_.moe.expert_overlay_participant_residency != nullptr;
                            sparse_dispatch_params.inbound_rows_lifetime = inbound_lifetime;
                            if (publishes_dispatch_payload)
                            {
                                /*
                                 * Ticket ownership follows this exact sparse
                                 * edge, not the graph-wide authority.  A
                                 * rank-local heterogeneous graph models every
                                 * logical source in one ordered collective;
                                 * non-root sources publish authenticated empty
                                 * packets and must never observe or retain the
                                 * root's captured payload ticket.
                                 */
                                sparse_dispatch_params.ticket_storage =
                                    dispatch_ticket_storage;
                                if (!dispatch_ticket_storage)
                                {
                                    sparse_dispatch_params.hidden = buffers.normalized;
                                    sparse_dispatch_params.routing_indices = routing_indices;
                                    sparse_dispatch_params.routing_weights = routing_weights;
                                    sparse_dispatch_params.hidden_buffer_id = buffers.idFor(BufferId::NORMALIZED);
                                    sparse_dispatch_params.routing_indices_buffer_id = buffers.idFor(BufferId::MOE_EXPERT_INDICES);
                                    sparse_dispatch_params.routing_weights_buffer_id = buffers.idFor(BufferId::MOE_EXPERT_WEIGHTS);
                                }
                                sparse_dispatch_params.dispatch_output_lifetime = dispatch_output_lifetime;
                                if (dispatch_ticket_storage)
                                {
                                    sparse_dispatch_params
                                        .ticket_observation_role =
                                        MoESparseDispatchStage::
                                            TicketObservationRole::
                                                MaterializedHostDispatch;
                                }
                            }

                            const std::string sparse_dispatch_name = prefix + "moe_sparse_dispatch_" +
                                                                     participant_suffix +
                                                                     "_from_p" + std::to_string(source_participant);
                            graph.addNode(sparse_dispatch_name,
                                          ComputeStageFactory::createMoESparseDispatch(sparse_dispatch_params),
                                          DeviceId::cpu());
                            if (cpu_llep_state)
                            {
                                cpu_llep_sparse_dispatch_nodes.push_back(
                                    sparse_dispatch_name);
                            }
                            graph.addDependency(sparse_dispatch_name,
                                                previous_dispatch_node.empty()
                                                    ? dispatch_dependency
                                                    : previous_dispatch_node);
                            /*
                             * MPI collectives form one globally ordered
                             * protocol across independently materialized rank
                             * graphs.  Make that order a graph edge rather than
                             * relying on unordered topological-sort tie breaks:
                             * every participant completes the previous target's
                             * return before entering the next dispatch.
                             */
                            if (!last_return_reduce.empty())
                                graph.addDependency(
                                    sparse_dispatch_name,
                                    last_return_reduce);
                            previous_dispatch_node = sparse_dispatch_name;
                            if (distributed_overlay ||
                                source_participant == target_participant)
                                target_dispatch_node = sparse_dispatch_name;
                        }

                        auto local_output_lifetime = std::make_shared<MoEOverlayReturnRows>(
                            participant_workspaces[static_cast<size_t>(target_participant)]->localExpertOutput(
                                layer_idx,
                                static_cast<int>(tier_index)));

                        std::shared_ptr<
                            MoEOverlayCanonicalRouteReturnTicketStorage>
                            canonical_route_ticket_storage;

                        const bool owns_target_participant =
                            !distributed_overlay ||
                            owns_overlay_participant(
                                target_participant);
                        std::string local_name;
                        if (owns_target_participant)
                        {
                            MoELocalExpertStage::Params local_params;
                            local_params.device_id = target_device;
                            local_params.input_rows_lifetime = target_dispatch_inbound;
                            local_params.output_rows_lifetime = local_output_lifetime;
                            local_params.workspace_lifetime =
                                participant_workspaces[static_cast<size_t>(
                                    target_participant)];
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
                            if (target_device.is_gpu() &&
                                config_.moe.rebalance_config.mode ==
                                    MoERebalanceRuntimeMode::Dynamic)
                            {
                                /*
                                 * The retained sparse child executes from its
                                 * epoch-indexed prepared residency bank. Give
                                 * it only the canonical table's observation
                                 * capability: lending the complete runtime
                                 * table here would create a second placement
                                 * authority inside the child executor.
                                 */
                                if (target_device != device ||
                                    !moe_runtime_table)
                                {
                                    throw std::logic_error(
                                        "Qwen35 MoE Dynamic sparse GPU endpoint has no graph-local canonical runtime table for participant " +
                                        std::to_string(target_participant));
                                }
                                local_params.overlay_service_telemetry =
                                    moe_runtime_table
                                        ->deviceOverlayServiceTelemetryBinding(
                                            layer_idx);
                                /* Unsampled exact-equivalent layers deliberately
                                 * bind no marker pair. The shared economy catalog
                                 * pools the bounded stratified sample before
                                 * expanding its service cost during certification. */
                            }
                            /*
                             * This participant-local child belongs to one
                             * immutable graph family. Name its economic phase
                             * from that graph's semantic role, never from M:
                             * fixed-depth MTP drafts and prefix-restore
                             * conditions are one-row graphs but remain routed
                             * work inside an MTP transaction.
                             */
                            local_params.service_phase = routed_service_phase;
                            /*
                             * The host packet arena is capacity-wide and shared
                             * by the serial graph family. Compact device tensors
                             * must still select this graph's live row bucket;
                             * inheriting the host arena capacity would make a
                             * one-token decode graph pay the prefill allocation
                             * and coherence footprint.
                             */
                            const size_t compact_row_capacity =
                                static_cast<size_t>(
                                    overlaySparseGraphRowCapacity(
                                        config_,
                                        target_device,
                                        total_tokens));
                            local_params.graph_row_capacity =
                                compact_row_capacity;
                            local_params.serial_compact_buffer_arena =
                                localExpertSerialBufferArenaForParticipant(
                                    target_device,
                                    target_participant,
                                    std::max<size_t>(
                                        compact_row_capacity,
                                        1u),
                                    captured_overlay_continuation &&
                                            direct_rank_local_protocol &&
                                            target_device.is_cpu()
                                        ? std::optional<DeviceId>{device}
                                        : std::nullopt);
                            if (captured_overlay_continuation &&
                                direct_rank_local_protocol &&
                                target_device.is_cpu())
                            {
                                if (!canonical_route_contributions ||
                                    !device.is_gpu())
                                {
                                    throw std::logic_error(
                                        "Qwen35 MoE colocated CPU endpoint has no continuation canonical-route bank");
                                }
                                auto contribution_region =
                                    local_params.serial_compact_buffer_arena
                                        ->mappedCPUCanonicalRoutes(device);
                                canonical_route_ticket_storage =
                                    std::make_shared<
                                        MoEOverlayCanonicalRouteReturnTicketStorage>();
                                canonical_route_ticket_storage
                                    ->bindFixedCapacity(
                                        layer_idx,
                                        static_cast<size_t>(total_tokens) *
                                            static_cast<size_t>(
                                                config_.moe.top_k),
                                        config_.d_model,
                                        device,
                                        /*workspace_generation=*/1u,
                                        std::move(contribution_region),
                                        mappedOverlayTicketArenaForDevice(
                                            device));
                                local_params
                                    .cpu_canonical_route_ticket_return =
                                    MoELocalExpertStage::
                                        CPUCanonicalRouteTicketReturnBinding{
                                            .storage =
                                                canonical_route_ticket_storage,
                                        };
                                has_rank_local_canonical_ticket_returns = true;
                            }
                            local_params.expert_weight_resolution_policy =
                                MoELocalExpertStage::ExpertWeightResolutionPolicy::
                                    RegistryOnly;
                            local_params.runtime_publication_stream =
                                target_device.is_gpu()
                                    ? device_state_publication_stream
                                    : nullptr;
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
                        /*
                         * A mapped endpoint is graph topology even when the
                         * automatic capacity plan gives it no epoch-one live
                         * experts. Its residency bank and compact arena are
                         * already preallocated; a later migration publishes
                         * engines through that bank. Validate prepared engines
                         * only for experts that are live in the initial mask.
                         */
                        if (hasActiveExpertMask(local_params.expert_mask) &&
                            !MoELocalExpertStage::prepareExpertGemmEngines(
                                local_params))
                        {
                            throw std::runtime_error(
                                "Qwen35 MoE graph failed to prepare participant-local "
                                "expert engines for layer " +
                                std::to_string(layer_idx) + " participant " +
                                std::to_string(target_participant) + " on " +
                                target_device.to_string());
                        }

                        if (config_.moe.expert_overlay_participant_residency)
                        {
                            if (!local_params.expert_registry)
                            {
                                throw std::runtime_error(
                                    "Qwen35 MoE production ExpertOverlay graph "
                                    "has no model-owned prepared engine registry");
                            }
                            const auto *participant =
                                owner_map_lifetime->participantForId(
                                    target_participant);
                            auto endpoint =
                                config_.moe
                                    .expert_overlay_participant_residency
                                    ->endpoint(target_participant);
                            if (!participant || !endpoint)
                            {
                                throw std::runtime_error(
                                    "Qwen35 MoE graph owns participant p" +
                                    std::to_string(target_participant) +
                                    " but its process-local residency endpoint is missing");
                            }

                            std::vector<
                                MoEOverlayPreparedExpertTriplet>
                                prepared_triplets;
                            std::string residency_error;
                            if (!resolveMoEOverlayPreparedExpertTriplets(
                                    *local_params.expert_registry,
                                    *participant,
                                    layer_idx,
                                    config_.moe.num_experts,
                                    local_params.expert_mask,
                                    prepared_triplets,
                                    &residency_error) ||
                                !config_.moe
                                     .expert_overlay_participant_residency
                                     ->registerInitialLayer(
                                         target_participant,
                                         layer_idx,
                                         local_params.expert_mask,
                                         prepared_triplets,
                                         &residency_error))
                            {
                                throw std::runtime_error(
                                    "Qwen35 MoE graph could not install initial "
                                    "prepared residency bank for layer " +
                                    std::to_string(layer_idx) + " participant " +
                                    std::to_string(target_participant) + ": " +
                                    residency_error);
                            }
                            local_params.overlay_participant_residency =
                                std::move(endpoint);
                        }
                        local_params.cpu_current_batch_llep_state =
                            cpu_llep_state;

                        local_name = prefix + "moe_local_expert_" + participant_suffix;
                        auto local_stage =
                            ComputeStageFactory::createMoELocalExpert(
                                local_params);
                        auto *concrete_local_stage =
                            dynamic_cast<MoELocalExpertStage *>(
                                local_stage.get());
                        if (!concrete_local_stage)
                        {
                            throw std::logic_error(
                                "Qwen35 MoE local expert factory returned an incompatible stage");
                        }
                        if (cpu_llep_state)
                        {
                            if (cpu_llep_local_consumer)
                            {
                                throw std::logic_error(
                                    "Qwen35 MoE CPU LLEP rank graph owns more than one local sparse endpoint");
                            }
                            cpu_llep_local_consumer =
                                concrete_local_stage;
                        }
                        if (target_device.is_gpu())
                        {
                            if (!device_state_publication_stream)
                            {
                                throw std::runtime_error(
                                    "Qwen35 MoE graph-native GPU participant "
                                    "requires an exact model-lifetime stream");
                            }
                            local_stage->setGPUStream(
                                device_state_publication_stream);
                            if (!concrete_local_stage
                                     ->preparePersistentBuffers())
                            {
                                throw std::runtime_error(
                                    "Qwen35 MoE graph-native GPU participant "
                                    "failed persistent sparse-buffer preparation");
                            }
                        }
                        graph.addNode(
                            local_name,
                            std::move(local_stage),
                            target_device);
                        graph.addDependency(local_name,
                                            target_dispatch_node.empty() ? previous_dispatch_node : target_dispatch_node);
                        }

                        const MoEOverlayCollectiveKey return_key = graphNativeMoEKey(
                            layer_idx,
                            static_cast<int>(tier_index),
                            target_participant,
                            MoEOverlayCollectiveDirection::ReturnReduce,
                            mtp_sidecar_context,
                            mtp_depth_idx);
                        std::string previous_return_node = last_return_reduce;
                        std::string root_return_node;
                        const std::vector<int> return_sources =
                            distributed_overlay || direct_rank_local_protocol
                                ? std::vector<int>{target_participant}
                                : participantsWithLast(
                                      participant_count,
                                      continuation_root_participant);
                        for (const int source_participant : return_sources)
                        {
                            std::shared_ptr<const MoEOverlayReturnRows> outbound_lifetime;
                            if (distributed_overlay ||
                                direct_rank_local_protocol ||
                                source_participant == target_participant)
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
                            if (canonical_route_ticket_storage)
                            {
                                return_params.inbound_consumer_role =
                                    MoESparseReturnReduceStage::
                                        InboundConsumerRole::
                                            CanonicalRouteTicketCompletion;
                                return_params
                                    .canonical_route_ticket_storage =
                                    canonical_route_ticket_storage;
                            }
                            else
                            {
                                return_params.ticket_storage =
                                    dispatch_ticket_storage;
                            }
                            if (!dispatch_ticket_storage &&
                                !canonical_route_ticket_storage)
                            {
                                return_params.dense_output = moe_output;
                                return_params.dense_output_buffer_id =
                                    buffers.idFor(
                                        BufferId::MOE_COMBINED_OUTPUT);
                            }
                            return_params.seq_len = total_tokens;
                            return_params.d_model = config_.d_model;
                            return_params.clear_output_before_scatter =
                                !canonical_route_ticket_storage &&
                                sparse_graph_contract
                                    .ownsDispatchAuthority() &&
                                first_return_scatter;
                            return_params.manual_boundary_requires_collective_completion =
                                distributed_overlay ||
                                direct_rank_local_protocol ||
                                source_participant ==
                                    continuation_root_participant;
                            return_params.require_explicit_transaction_identity =
                                distributed_overlay;
                            return_params.require_explicit_execution_semantics =
                                config_.moe.expert_overlay_participant_residency != nullptr;
                            const bool is_final_ordered_return =
                                sparse_graph_contract
                                    .ownsDispatchAuthority() &&
                                final_host_sparse_return.has_value() &&
                                final_host_sparse_return->matches(
                                    tier_index,
                                    target_participant) &&
                                !return_sources.empty() &&
                                source_participant == return_sources.back();
                            return_params.publish_ticket_completion =
                                dispatch_ticket_storage &&
                                !canonical_route_ticket_storage &&
                                is_final_ordered_return;
                            return_params.residency_lease_terminal =
                                moeOverlayHostDispatchLeaseTerminal(
                                    host_dispatch_lease_owner,
                                    is_final_ordered_return);
                            if (return_params.residency_lease_terminal ==
                                MoEOverlayHostDispatchLeaseTerminal::Release)
                            {
                                return_params.dispatch_output_lifetime =
                                    dispatch_output_lifetime;
                            }

                            const std::string return_name = prefix + "moe_sparse_return_reduce_" +
                                                            participant_suffix +
                                                            "_from_p" + std::to_string(source_participant);
                            graph.addNode(return_name,
                                          ComputeStageFactory::createMoESparseReturnReduce(return_params),
                                          DeviceId::cpu());
                            graph.addDependency(
                                return_name,
                                local_name.empty()
                                    ? (target_dispatch_node.empty()
                                           ? previous_dispatch_node
                                           : target_dispatch_node)
                                    : local_name);
                            if (!previous_return_node.empty())
                                graph.addDependency(return_name, previous_return_node);
                            previous_return_node = return_name;
                            if (distributed_overlay ||
                                direct_rank_local_protocol ||
                                source_participant ==
                                    continuation_root_participant)
                                root_return_node = return_name;
                        }

                        if (!root_return_node.empty())
                        {
                            std::string tier_terminal = root_return_node;
                            const bool final_ticket_return =
                                final_host_sparse_return.has_value() &&
                                final_host_sparse_return->matches(
                                    tier_index,
                                    target_participant);
                            if (canonical_route_ticket_storage)
                            {
                                /* Defer every GPU ingress until all colocated
                                 * CPU endpoints have completed. This creates
                                 * one manual fan-in followed by one captured
                                 * ingress wave instead of alternating capture
                                 * and host execution per participant. */
                                pending_canonical_ticket_consumes.push_back(
                                    PendingCanonicalTicketConsume{
                                        .participant_suffix =
                                            participant_suffix,
                                        .storage =
                                            canonical_route_ticket_storage,
                                    });
                            }
                            else if (dispatch_ticket_storage &&
                                     final_ticket_return)
                            {
                                MoEOverlayTicketConsumeStage::Params
                                    consume_params;
                                consume_params.device_id = device;
                                consume_params.output =
                                    captured_overlay_continuation
                                        ? buffers.attn_proj
                                        : moe_output;
                                consume_params.output_buffer_id =
                                    captured_overlay_continuation
                                        ? buffers.idFor(BufferId::ATTN_PROJ)
                                        : buffers.idFor(
                                              BufferId::MOE_COMBINED_OUTPUT);
                                consume_params.layer_idx = layer_idx;
                                consume_params.bucket_rows = total_tokens;
                                consume_params.d_model = config_.d_model;
                                consume_params.ticket_storage =
                                    dispatch_ticket_storage;

                                const std::string consume_name =
                                    prefix +
                                    "moe_overlay_ticket_consume_" +
                                    participant_suffix;
                                graph.addNode(
                                    consume_name,
                                    ComputeStageFactory::
                                        createMoEOverlayTicketConsume(
                                            consume_params),
                                    device);
                                graph.addDependency(
                                    consume_name,
                                    root_return_node);
                                if (captured_overlay_continuation)
                                {
                                    graph.setGraphCaptureWaveContract(
                                        consume_name,
                                        GraphCaptureWaveContract{
                                            .identity =
                                                captured_overlay_inbound_wave,
                                        });
                                }
                                tier_terminal = consume_name;
                            }
                            if (compute_apportioned_tier_on_graph_local_participant && needsMoEParticipantAllreduce())
                            {
                                const size_t allreduce_count =
                                    static_cast<size_t>(total_tokens) * static_cast<size_t>(config_.d_model);
                                const std::string ar_name = prefix + "moe_sparse_return_reduce_" +
                                                            nodeSuffixForTier(tier, static_cast<int>(tier_index)) +
                                                            "_allreduce";
                                auto rebalance_sidebands = takeGraphRebalanceSidebandsForAllreduce();
                                auto allreduce_stage = createTPAllreduceStage(
                                    moe_output,
                                    allreduce_count,
                                    device,
                                    layer_idx,
                                    /*is_attention=*/false,
                                    ar_name,
                                    buffers.idFor(BufferId::MOE_COMBINED_OUTPUT),
                                    std::move(rebalance_sidebands));
                                if (allreduce_stage)
                                {
                                    graph.addNode(ar_name, std::move(allreduce_stage), device);
                                    graph.addDependency(ar_name, tier_terminal);
                                    tier_terminal = ar_name;
                                    maybeAddGraphRebalancePayloadStageAfterSideband(
                                        ar_name,
                                        tier_terminal);
                                }
                            }

                            last_return_reduce = tier_terminal;
                            first_return_scatter = false;
                        }
                    }
                }

                if (!pending_canonical_ticket_consumes.empty())
                {
                    if (last_return_reduce.empty())
                    {
                        throw std::logic_error(
                            "Qwen35 MoE canonical CPU ticket ingress has no completed manual frontier");
                    }
                    std::string ingress_dependency = last_return_reduce;
                    for (const auto &pending :
                         pending_canonical_ticket_consumes)
                    {
                        MoEOverlayTicketConsumeStage::Params consume_params;
                        consume_params.device_id = device;
                        consume_params.output = canonical_route_contributions;
                        consume_params.output_buffer_id = buffers.idFor(
                            BufferId::MOE_CANONICAL_ROUTE_CONTRIBUTIONS);
                        consume_params.layer_idx = layer_idx;
                        consume_params.bucket_rows = total_tokens;
                        consume_params.top_k = config_.moe.top_k;
                        consume_params.d_model = config_.d_model;
                        consume_params.canonical_route_ticket_storage =
                            pending.storage;

                        const std::string consume_name =
                            prefix +
                            "moe_overlay_canonical_ticket_consume_" +
                            pending.participant_suffix;
                        graph.addNode(
                            consume_name,
                            ComputeStageFactory::
                                createMoEOverlayTicketConsume(
                                    consume_params),
                            device);
                        graph.addDependency(
                            consume_name, ingress_dependency);
                        graph.setGraphCaptureWaveContract(
                            consume_name,
                            GraphCaptureWaveContract{
                                .identity = captured_overlay_inbound_wave,
                            });
                        ingress_dependency = consume_name;
                    }
                    local_canonical_ticket_terminal = ingress_dependency;
                    last_return_reduce = ingress_dependency;
                }

                const std::string rank_local_ticket_return_terminal =
                    rank_local_collective_context_lifetime &&
                            !has_rank_local_canonical_ticket_returns
                        ? last_return_reduce
                        : std::string{};

                if (use_mapped_activation_parent)
                {
                    if (mapped_activation_return_bindings.empty() ||
                        captured_local_expert_terminal.empty())
                    {
                        throw std::runtime_error(
                            "Qwen35 MoE mapped activation parent has an incomplete fanout or continuation-local overlap body");
                    }

                    /*
                     * Remote lanes are independent until their returned values
                     * enter the continuation accumulator. Exact one-row direct
                     * mappings use one topology-sized kernel grid. Wider rows
                     * use one typed fork/join transaction: every dispatch and
                     * return acquisition runs on a persistent auxiliary lane
                     * while continuation-local experts run on the public graph
                     * stream. The join folds lanes in this binding vector's
                     * planner-canonical order, so completion timing cannot
                     * perturb FP32 arithmetic.
                     */
                    const bool use_one_row_lane_batch =
                        total_tokens == 1 &&
                        mapped_activation_return_bindings.size() > 1u &&
                        std::all_of(
                            mapped_activation_return_bindings.begin(),
                            mapped_activation_return_bindings.end(),
                            [](const MappedActivationReturnBinding &binding)
                            {
                                const auto payload =
                                    binding.lane.dispatchPayload(1);
                                return payload.valid() &&
                                       payload.selection.usesCompactRows();
                            });

                    const bool use_asynchronous_lane_batch =
                        total_tokens > 1;
                    const bool use_lane_batch =
                        use_one_row_lane_batch ||
                        use_asynchronous_lane_batch;
                    std::shared_ptr<
                        MoEOverlayActivationLaneBatchState>
                        mapped_lane_transaction;
                    if (use_lane_batch)
                    {
                        std::vector<
                            MoEOverlayMappedActivationDeviceLane>
                            lanes;
                        lanes.reserve(
                            mapped_activation_return_bindings.size());
                        for (const auto &binding :
                             mapped_activation_return_bindings)
                        {
                            lanes.push_back(binding.lane);
                        }
                        mapped_lane_transaction = std::make_shared<
                            MoEOverlayActivationLaneBatchState>(
                            device,
                            std::move(lanes),
                            total_tokens,
                            mapped_activation_return_bindings.front()
                                .stage_ordinal,
                            layer_idx);
                    }

                    if (use_lane_batch)
                    {
                        MoEOverlayActivationDispatchPackBatchStage::Params
                            packet_params;
                        packet_params.device_id = device;
                        packet_params.transaction =
                            mapped_lane_transaction;
                        packet_params.hidden = buffers.normalized;
                        packet_params.routing_indices = routing_indices;
                        packet_params.routing_weights = routing_weights;
                        packet_params.placement = pinned_route_placement;
                        /* A sequence-length scalar is also the live-row count
                         * only for a padded single-request prefill bucket.
                         * During decode it contains the growing KV position,
                         * which can be hundreds of rows while this retained
                         * graph owns exactly one activation row. Binding it to
                         * decode would make every sparse packet fail its
                         * capacity contract after a non-empty prefill. */
                        packet_params.active_row_count_device =
                            batch_size == 1 && total_tokens > 1
                                ? sequence_lengths_device
                                : nullptr;
                        packet_params.hidden_buffer_id =
                            buffers.idFor(BufferId::NORMALIZED);
                        packet_params.routing_indices_buffer_id =
                            buffers.idFor(BufferId::MOE_EXPERT_INDICES);
                        packet_params.routing_weights_buffer_id =
                            buffers.idFor(BufferId::MOE_EXPERT_WEIGHTS);
                        const std::string packet_name =
                            prefix +
                            "moe_overlay_activation_dispatch_pack_batch";
                        graph.addNode(
                            packet_name,
                            ComputeStageFactory::
                                createMoEOverlayActivationDispatchPackBatch(
                                    packet_params),
                            device);
                        GraphCaptureWaveContract packet_wave{
                            .identity = mapped_dispatch_wave_identity(
                                mapped_lane_transaction->lanes().front()
                                    .target_participant_id),
                        };
                        for (std::size_t lane = 1u;
                             lane < mapped_lane_transaction->lanes().size();
                             ++lane)
                        {
                            packet_wave.passive_following_identities.push_back(
                                mapped_dispatch_wave_identity(
                                    mapped_lane_transaction->lanes()[lane]
                                        .target_participant_id));
                        }
                        graph.setGraphCaptureWaveContract(
                            packet_name, std::move(packet_wave));
                        graph.addDependency(
                            packet_name, prefix + "moe_routing");
                        mapped_activation_dispatch_nodes.push_back(packet_name);
                    }
                    else
                    {
                        for (const auto &binding :
                             mapped_activation_return_bindings)
                        {
                            MoEOverlayActivationDispatchPackStage::Params
                                packet_params;
                            packet_params.device_id = device;
                            packet_params.lane = binding.lane;
                            packet_params.hidden = buffers.normalized;
                            packet_params.routing_indices = routing_indices;
                            packet_params.routing_weights = routing_weights;
                            packet_params.placement = pinned_route_placement;
                            /* Decode has one live activation row regardless of
                             * the current sequence/KV length. Only padded
                             * prefill buckets may derive live rows from the
                             * sequence-length scalar. */
                            packet_params.active_row_count_device =
                                batch_size == 1 && total_tokens > 1
                                    ? sequence_lengths_device
                                    : nullptr;
                            packet_params.hidden_buffer_id =
                                buffers.idFor(BufferId::NORMALIZED);
                            packet_params.routing_indices_buffer_id =
                                buffers.idFor(
                                    BufferId::MOE_EXPERT_INDICES);
                            packet_params.routing_weights_buffer_id =
                                buffers.idFor(BufferId::MOE_EXPERT_WEIGHTS);
                            packet_params.physical_rows = total_tokens;
                            packet_params.stage_ordinal =
                                binding.stage_ordinal;
                            packet_params.model_layer_index = layer_idx;

                            graph.addNode(
                                binding.dispatch_node_name,
                                ComputeStageFactory::
                                    createMoEOverlayActivationDispatchPack(
                                        packet_params),
                                device);
                            graph.setGraphCaptureWaveContract(
                                binding.dispatch_node_name,
                                GraphCaptureWaveContract{
                                    .identity =
                                        mapped_dispatch_wave_identity(
                                            binding.lane
                                                .target_participant_id),
                                });
                            graph.addDependency(
                                binding.dispatch_node_name,
                                mapped_activation_dispatch_nodes.empty()
                                    ? prefix + "moe_routing"
                                    : mapped_activation_dispatch_nodes.back());
                            mapped_activation_dispatch_nodes.push_back(
                                binding.dispatch_node_name);
                        }
                    }

                    if (mapped_activation_dispatch_nodes.empty())
                    {
                        throw std::runtime_error(
                            "Qwen35 MoE mapped activation parent produced no dispatch stage");
                    }
                    graph.addDependency(
                        captured_local_expert_compute_node,
                        mapped_activation_dispatch_nodes.back());

                    if (use_lane_batch)
                    {
                        MoEOverlayActivationReturnConsumeBatchStage::Params
                            return_params;
                        return_params.device_id = device;
                        return_params.transaction =
                            mapped_lane_transaction;
                        return_params.canonical_route_contributions =
                            canonical_route_contributions;
                        return_params
                            .canonical_route_contributions_buffer_id =
                            buffers.idFor(
                                BufferId::MOE_CANONICAL_ROUTE_CONTRIBUTIONS);
                        const std::string return_name =
                            prefix +
                            "moe_overlay_activation_return_consume_batch";
                        graph.addNode(
                            return_name,
                            ComputeStageFactory::
                                createMoEOverlayActivationReturnConsumeBatch(
                                    return_params),
                            device);
                        GraphCaptureWaveContract return_wave{
                            .identity = mapped_return_wave_identity(
                                mapped_lane_transaction->lanes().front()
                                    .target_participant_id),
                        };
                        for (std::size_t lane = 1u;
                             lane < mapped_lane_transaction->lanes().size();
                             ++lane)
                        {
                            return_wave.passive_following_identities.push_back(
                                mapped_return_wave_identity(
                                    mapped_lane_transaction->lanes()[lane]
                                        .target_participant_id));
                        }
                        graph.setGraphCaptureWaveContract(
                            return_name, std::move(return_wave));
                        graph.addDependency(
                            return_name, captured_local_expert_terminal);
                        if (!local_canonical_ticket_terminal.empty())
                        {
                            graph.addDependency(
                                return_name,
                                local_canonical_ticket_terminal);
                        }
                        mapped_activation_return_nodes.push_back(
                            return_name);
                    }
                    else
                    {
                        for (const auto &binding :
                             mapped_activation_return_bindings)
                        {
                            MoEOverlayActivationReturnConsumeStage::Params
                                return_params;
                            return_params.device_id = device;
                            return_params.lane = binding.lane;
                            return_params.canonical_route_contributions =
                                canonical_route_contributions;
                            return_params
                                .canonical_route_contributions_buffer_id =
                                buffers.idFor(
                                    BufferId::
                                        MOE_CANONICAL_ROUTE_CONTRIBUTIONS);
                            return_params.physical_rows = total_tokens;
                            return_params.stage_ordinal =
                                binding.stage_ordinal;
                            return_params.model_layer_index = layer_idx;
                            graph.addNode(
                                binding.node_name,
                                ComputeStageFactory::
                                    createMoEOverlayActivationReturnConsume(
                                        return_params),
                                device);
                            graph.setGraphCaptureWaveContract(
                                binding.node_name,
                                GraphCaptureWaveContract{
                                    .identity =
                                        mapped_return_wave_identity(
                                            return_params.lane
                                                .target_participant_id),
                                });
                            graph.addDependency(
                                binding.node_name,
                                mapped_activation_return_nodes.empty()
                                    ? captured_local_expert_terminal
                                    : mapped_activation_return_nodes.back());
                            if (mapped_activation_return_nodes.empty() &&
                                !local_canonical_ticket_terminal.empty())
                            {
                                graph.addDependency(
                                    binding.node_name,
                                    local_canonical_ticket_terminal);
                            }
                            mapped_activation_return_nodes.push_back(
                                binding.node_name);
                        }
                    }
                    last_return_reduce =
                        mapped_activation_return_nodes.back();
                    first_return_scatter = false;

                }

                if (last_return_reduce.empty() &&
                    !captured_overlay_continuation)
                {
                    throw std::runtime_error("Qwen35 MoE graph-native overlay produced no local expert participants for layer " +
                                             std::to_string(layer_idx));
                }
                if (captured_overlay_continuation &&
                    captured_local_expert_terminal.empty())
                {
                    throw std::runtime_error(
                        "Qwen35 MoE distributed GPU continuation omitted its graph-local captured expert branch");
                }

                if (cpu_llep_state)
                {
                    if (!cpu_llep_overlay_domain || !global_tp_ctx ||
                        !cpu_llep_local_consumer ||
                        global_tp_ctx->degree() != participant_count ||
                        global_tp_ctx->myIndex() !=
                            cpu_llep_local_consumer
                                ->cpuCurrentBatchLLEPParticipantId())
                    {
                        throw std::logic_error(
                            "Qwen35 MoE CPU LLEP sparse graph has no exact local participant endpoint");
                    }

                    MoECPUCurrentBatchLLEPStage::Params begin_params;
                    begin_params.device_id = device;
                    begin_params.phase =
                        CPUCurrentBatchLLEPPhase::Begin;
                    begin_params.physical_executor =
                        cpu_current_batch_llep_executor_;
                    begin_params.residency_authority =
                        config_.moe.expert_overlay_residency_authority;
                    begin_params.residency_domain =
                        cpu_llep_overlay_domain->name;
                    begin_params.tp_ctx = global_tp_ctx;
                    begin_params.expert_consumer =
                        cpu_llep_local_consumer;
                    begin_params.state = cpu_llep_state;
                    begin_params.routing_indices = routing_indices;
                    begin_params.routing_weights = routing_weights;
                    begin_params.routing_indices_buffer_id =
                        buffers.idFor(BufferId::MOE_EXPERT_INDICES);
                    begin_params.routing_weights_buffer_id =
                        buffers.idFor(BufferId::MOE_EXPERT_WEIGHTS);
                    begin_params.layer_idx = layer_idx;
                    begin_params.seq_len = total_tokens;
                    begin_params.top_k = config_.moe.top_k;
                    begin_params.num_experts = config_.moe.num_experts;
                    begin_params.participant_count = participant_count;
                    begin_params.participant_id =
                        global_tp_ctx->myIndex();
                    begin_params.planner_config.alpha_numerator =
                        std::max<uint32_t>(
                            1u,
                            config_.moe.routed_prefill_config
                                .llep_alpha_numerator);
                    begin_params.planner_config.alpha_denominator =
                        std::max<uint32_t>(
                            1u,
                            config_.moe.routed_prefill_config
                                .llep_alpha_denominator);
                    begin_params.planner_config.lambda_numerator =
                        std::max<uint32_t>(
                            1u,
                            config_.moe.routed_prefill_config
                                .llep_lambda_numerator);
                    begin_params.planner_config.lambda_denominator =
                        std::max<uint32_t>(
                            1u,
                            config_.moe.routed_prefill_config
                                .llep_lambda_denominator);
                    begin_params.planner_config.enable_balanced_skip =
                        config_.moe.routed_prefill_config
                            .llep_enable_balanced_skip;
                    begin_params.stage_name =
                        prefix + "moe_cpu_current_batch_llep_begin";
                    const std::string begin_name =
                        begin_params.stage_name;
                    graph.addNode(
                        begin_name,
                        ComputeStageFactory::createMoECPUCurrentBatchLLEP(
                            begin_params),
                        device);
                    graph.addDependency(
                        begin_name, prefix + "moe_routing");
                    if (sparse_graph_contract.ownsDispatchAuthority())
                        graph.addDependency(dispatch_name, begin_name);
                    for (const auto &sparse_dispatch_name :
                         cpu_llep_sparse_dispatch_nodes)
                    {
                        graph.addDependency(
                            sparse_dispatch_name, begin_name);
                    }

                    MoECPUCurrentBatchLLEPStage::Params restore_params;
                    restore_params.device_id = device;
                    restore_params.phase =
                        CPUCurrentBatchLLEPPhase::Restore;
                    restore_params.physical_executor =
                        cpu_current_batch_llep_executor_;
                    restore_params.residency_authority =
                        config_.moe.expert_overlay_residency_authority;
                    restore_params.residency_domain =
                        cpu_llep_overlay_domain->name;
                    restore_params.tp_ctx = global_tp_ctx;
                    restore_params.expert_consumer =
                        cpu_llep_local_consumer;
                    restore_params.state = cpu_llep_state;
                    restore_params.layer_idx = layer_idx;
                    restore_params.seq_len = total_tokens;
                    restore_params.top_k = config_.moe.top_k;
                    restore_params.num_experts = config_.moe.num_experts;
                    restore_params.participant_count = participant_count;
                    restore_params.participant_id =
                        global_tp_ctx->myIndex();
                    restore_params.stage_name =
                        prefix + "moe_cpu_current_batch_llep_restore";
                    const std::string restore_name =
                        restore_params.stage_name;
                    graph.addNode(
                        restore_name,
                        ComputeStageFactory::createMoECPUCurrentBatchLLEP(
                            restore_params),
                        device);
                    graph.addDependency(
                        restore_name, last_return_reduce);
                    last_return_reduce = restore_name;
                }

                /*
                 * Sparse return has one logical continuation owner: only that
                 * participant's MOE_COMBINED_OUTPUT contains the complete
                 * routed row.  A NodeTP continuation, however, resumes dense
                 * execution on every rank.  Publish the compact active prefix
                 * through the continuation domain before any rank consumes it;
                 * otherwise the non-root shards feed stale routed activations
                 * into the next layer and the first later TP collective mixes
                 * different hidden states.
                 *
                 * This is a production graph edge, not parity reconstruction.
                 * Every rank materializes the same rooted collective node after
                 * the globally ordered sparse protocol.  Single-participant
                 * heterogeneous overlays retain their root-local result and do
                 * not pay for a redundant broadcast.
                 */
                const auto merge_direct_ticket_routes_after =
                    [&](const std::string &mapped_return_terminal,
                        bool annotate_inbound_capture_wave)
                {
                    if (rank_local_ticket_return_terminal.empty())
                        return mapped_return_terminal;

                    /*
                     * Mapped remote lanes accumulate directly into the
                     * continuation output while the colocated CPU endpoint
                     * accumulates into the fixed ticket scratch. Join the two
                     * independent branches once, after both exact publication
                     * edges, without making either producer wait for the
                     * other.
                     */
                    ResidualAddStage::Params merge_params;
                    merge_params.device_id = device;
                    merge_params.input = buffers.attn_proj;
                    merge_params.residual = moe_output;
                    merge_params.output = moe_output;
                    merge_params.num_elements =
                        static_cast<size_t>(total_tokens) *
                        static_cast<size_t>(config_.d_model);
                    merge_params.input_buffer_id =
                        buffers.idFor(BufferId::ATTN_PROJ);
                    merge_params.residual_buffer_id =
                        buffers.idFor(BufferId::MOE_COMBINED_OUTPUT);
                    merge_params.output_buffer_id =
                        buffers.idFor(BufferId::MOE_COMBINED_OUTPUT);
                    const std::string merge_name =
                        prefix + "moe_overlay_colocated_routes_merge";
                    graph.addNode(
                        merge_name,
                        ComputeStageFactory::createResidualAdd(merge_params),
                        device);
                    graph.addDependency(
                        merge_name, mapped_return_terminal);
                    graph.addDependency(
                        merge_name, rank_local_ticket_return_terminal);
                    if (annotate_inbound_capture_wave)
                    {
                        graph.setGraphCaptureWaveContract(
                            merge_name,
                            GraphCaptureWaveContract{
                                .identity = captured_overlay_inbound_wave,
                            });
                    }
                    return merge_name;
                };

                const bool distributed_dense_continuation =
                    distributed_overlay &&
                    overlay_plan->continuation_domain_spec
                            .effectiveDensePolicy() ==
                        DenseParallelPolicy::TensorParallel;
                if (captured_local_tp_continuation)
                {
                    if (!local_tp_ctx || continuation_root_tp_index < 0 ||
                        continuation_root_tp_index >= local_tp_ctx->degree() ||
                        config_.tp_device_idx < 0 ||
                        config_.tp_device_idx >= local_tp_ctx->degree() ||
                        !canonical_route_contributions)
                    {
                        throw std::runtime_error(
                            "Qwen35 MoE captured distributed LocalTP publication has an incomplete rooted collective contract");
                    }
                    if (sparse_graph_contract.ownsDispatchAuthority() &&
                        last_return_reduce.empty())
                    {
                        throw std::runtime_error(
                            "Qwen35 MoE captured distributed LocalTP root has no completed remote sparse return");
                    }

                    const auto route_transport =
                        config_.moe.node_local_route_transport;
                    if (route_transport ==
                        MoEOverlayNodeLocalRouteTransport::Unresolved)
                    {
                        throw std::runtime_error(
                            "Qwen35 MoE captured multi-GPU continuation has no resolved node-local route transport");
                    }
                    const bool use_mapped_sparse_routes =
                        route_transport ==
                        MoEOverlayNodeLocalRouteTransport::MappedSparse;
                    const auto &route_exchange =
                        config_.moe.node_local_route_exchange;
                    if (use_mapped_sparse_routes !=
                        static_cast<bool>(route_exchange))
                    {
                        throw std::runtime_error(
                            "Qwen35 MoE node-local route transport and mapped fabric ownership disagree");
                    }

                    MoEDomainRouteAssignmentLedger
                        domain_route_assignment{};
                    MoERuntimeRouteWeightBinding runtime_route_weights{};
                    MoEOverlayRoutePlacementDeviceBinding
                        overlay_route_placement{};
                    if (captured_distributed_overlay_runtime_table)
                    {
                        /*
                         * Route evidence follows the device-authored routing
                         * transaction, not the transport selected for canonical
                         * expert rows. Native NCCL/RCCL and mapped sparse lanes
                         * consume the same request-pinned placement epoch and
                         * final route schedule, so both must expose the same
                         * authenticated evidence at the root reducer boundary.
                         */
                        if (!moe_runtime_table || layer_idx < 0 ||
                            layer_idx >= moe_runtime_table->layerCount())
                        {
                            throw std::runtime_error(
                                "Qwen35 MoE captured LocalTP continuation requires a complete device runtime table");
                        }
                        const auto &runtime_layer =
                            moe_runtime_table->hostLayerState(layer_idx);
                        const std::uint64_t required_route_slots =
                            static_cast<std::uint64_t>(total_tokens) *
                            static_cast<std::uint64_t>(config_.moe.top_k);
                        if (!runtime_layer.route_participant_ids ||
                            required_route_slots == 0u ||
                            required_route_slots >
                                static_cast<std::uint64_t>(
                                    runtime_layer.prefill_route_capacity))
                        {
                            throw std::runtime_error(
                                "Qwen35 MoE captured LocalTP continuation has no complete final device assignment ledger");
                        }
                        domain_route_assignment = {
                            .participant_ids =
                                runtime_layer.route_participant_ids,
                            .capacity =
                                runtime_layer.prefill_route_capacity,
                        };
                        runtime_route_weights = bindMoERuntimeRouteWeights(
                            moe_runtime_table->deviceLayerState(layer_idx),
                            runtime_layer,
                            captured_overlay_route_weight_projection);
                        if (!runtime_route_weights.validFor(
                                static_cast<std::uint32_t>(total_tokens),
                                static_cast<std::uint32_t>(
                                    config_.moe.top_k)))
                        {
                            throw std::runtime_error(
                                "Qwen35 MoE captured LocalTP continuation has no workload-correct final device route-weight publication");
                        }
                        overlay_route_placement =
                            moe_runtime_table->overlayRoutePlacementBinding(
                                layer_idx);
                        if (!overlay_route_placement.valid())
                        {
                            throw std::runtime_error(
                                "Qwen35 MoE captured LocalTP continuation has no request-pinned global placement binding");
                        }
                    }
                    if (use_mapped_sparse_routes)
                    {
                        /*
                         * With no native P2P, each non-root publishes only the
                         * original route slots assigned to it. The root reads
                         * local VRAM or an exact mapped lane per slot and keeps
                         * the serial router order without transporting zeroes.
                         */
                        std::vector<MoENodeLocalRouteEndpoint> route_endpoints;
                        for (const auto &participant :
                             owner_map_lifetime->participants())
                        {
                            if (participant.tier_idx ==
                                    continuation_tier_index &&
                                continuation_expert_domain &&
                                participant.domain_name ==
                                    continuation_expert_domain->name)
                            {
                                route_endpoints.push_back({
                                    .participant_id =
                                        participant.domain_participant_index,
                                    .device = participant.device,
                                });
                            }
                        }
                        if (route_endpoints.size() !=
                            static_cast<std::size_t>(local_tp_ctx->degree()))
                        {
                            throw std::runtime_error(
                                "Qwen35 MoE sparse continuation route endpoint count differs from its LocalTP device cell");
                        }
                        route_exchange->materialize(
                            std::move(route_endpoints),
                            continuation_root_tp_index,
                            static_cast<std::uint32_t>(
                                graphStableActivationRowCapacity(
                                    config_, device)),
                            static_cast<std::uint32_t>(config_.moe.top_k),
                            static_cast<std::uint32_t>(config_.d_model));
                        /* Assignment and placement evidence was bound above
                         * from the common runtime authority. The sparse fabric
                         * adds transport storage only; it does not redefine
                         * routing truth. */
                    }

                    std::string ordered_reduce_dependency =
                        captured_local_expert_terminal;
                    if (route_transport ==
                        MoEOverlayNodeLocalRouteTransport::NativeCollective)
                    {
                        /*
                         * A P2P-capable homogeneous domain keeps NCCL/RCCL in
                         * charge of topology. Reduce the live canonical route
                         * prefix to the fixed continuation root, then perform
                         * the same ordered FP32 fold used by the mapped path.
                         * Non-root reducers remain declarative passive nodes.
                         */
                        TPLocalRootedCollectiveStage::Params rooted_params;
                        rooted_params.device_id = device;
                        rooted_params.tp_ctx = local_tp_ctx;
                        rooted_params.tensor = canonical_route_contributions;
                        rooted_params.count =
                            static_cast<std::size_t>(total_tokens) *
                            static_cast<std::size_t>(config_.moe.top_k) *
                            static_cast<std::size_t>(config_.d_model);
                        rooted_params.dtype = CollectiveDataType::FLOAT32;
                        rooted_params.operation =
                            TPLocalRootedCollectiveOperation::ReduceSum;
                        rooted_params.root_device_index =
                            continuation_root_tp_index;
                        rooted_params.participant_device_index =
                            config_.tp_device_idx;
                        rooted_params.stage_name =
                            prefix +
                            "moe_overlay_continuation_routes_reduce_to_root";
                        rooted_params.tensor_buffer_id = buffers.idFor(
                            BufferId::MOE_CANONICAL_ROUTE_CONTRIBUTIONS);
                        graph.addNode(
                            rooted_params.stage_name,
                            ComputeStageFactory::createTPLocalRootedCollective(
                                rooted_params),
                            device);
                        graph.addDependency(
                            rooted_params.stage_name,
                            captured_local_expert_terminal);
                        ordered_reduce_dependency = rooted_params.stage_name;
                    }

                    if (use_mapped_activation_parent)
                    {
                        if (mapped_activation_return_nodes.empty() ||
                            last_return_reduce.empty())
                        {
                            throw std::runtime_error(
                                "Qwen35 MoE mapped activation parent has no canonical return materialization frontier");
                        }
                        /* Native LocalTP reduction, when selected, owns the
                         * continuation-domain slots first. Mapped followers
                         * then fill disjoint external-domain slots. Exactly one
                         * ordered reducer consumes the completed canonical bank. */
                        graph.addDependency(
                            mapped_activation_return_nodes.front(),
                            ordered_reduce_dependency);
                        ordered_reduce_dependency = last_return_reduce;
                    }

                    MoECanonicalRouteReduceStage::Params reduce_params;
                    reduce_params.device_id = device;
                    reduce_params.canonical_route_contributions =
                        canonical_route_contributions;
                    reduce_params.output = moe_output;
                    reduce_params.seq_len = total_tokens;
                    reduce_params.top_k = config_.moe.top_k;
                    reduce_params.d_model = config_.d_model;
                    reduce_params.canonical_route_arithmetic =
                        MoECanonicalRouteArithmeticPolicy::
                            PreweightedContributionThenOrderedAdd;
                    reduce_params.canonical_route_layout =
                        MoECanonicalRoutePublicationLayout::
                            DenseOriginalRouteSlots;
                    reduce_params.reduction_role =
                        config_.tp_device_idx == continuation_root_tp_index
                            ? MoECanonicalRouteReductionRole::RootOwner
                            : MoECanonicalRouteReductionRole::
                                  NonRootParticipant;
                    reduce_params.node_local_route_exchange =
                        route_exchange;
                    reduce_params.domain_route_assignment =
                        domain_route_assignment;
                    reduce_params.external_route_source =
                        use_mapped_activation_parent ||
                                has_rank_local_canonical_ticket_returns
                            ? MoEExternalCanonicalRouteSource::
                                  RootCanonicalRouteBank
                            : MoEExternalCanonicalRouteSource::
                                  DeferredDenseMerge;
                    reduce_params.runtime_route_weights =
                        runtime_route_weights;
                    reduce_params.overlay_route_placement =
                        overlay_route_placement;
                    reduce_params.route_participant_id =
                        config_.tp_device_idx;
                    reduce_params.canonical_route_contributions_buffer_id =
                        buffers.idFor(
                            BufferId::MOE_CANONICAL_ROUTE_CONTRIBUTIONS);
                    reduce_params.output_buffer_id =
                        buffers.idFor(BufferId::MOE_COMBINED_OUTPUT);

                    const std::string ordered_reduce_name =
                        prefix +
                        "moe_overlay_continuation_routes_ordered_reduce";
                    graph.addNode(
                        ordered_reduce_name,
                        ComputeStageFactory::createMoECanonicalRouteReduce(
                            reduce_params),
                        device);
                    graph.addDependency(
                        ordered_reduce_name,
                        ordered_reduce_dependency);
                    if (!local_canonical_ticket_terminal.empty())
                    {
                        graph.addDependency(
                            ordered_reduce_name,
                            local_canonical_ticket_terminal);
                    }
                    captured_overlay_routed_unit_terminal =
                        ordered_reduce_name;
                    GraphCaptureWaveContract ordered_reduce_capture_wave{
                        .identity =
                            captured_overlay_outbound_wave,
                    };
                    if (!sparse_graph_contract
                             .ownsDispatchAuthority() &&
                        !mapped_activation_return_wave_identities.empty())
                    {
                        /* Root-only return consumers are captured after the
                         * common ordered reduction and before the common
                         * publication broadcast. Join their exact lane order
                         * passively so both LocalTP participants retain one
                         * capture-wave ordinal sequence. */
                        ordered_reduce_capture_wave
                            .passive_following_identities =
                            mapped_activation_return_wave_identities;
                    }
                    graph.setGraphCaptureWaveContract(
                        ordered_reduce_name,
                        std::move(ordered_reduce_capture_wave));
                    if (requiresHeterogeneousTicketSegmentation(
                            graph.nativeCaptureEnvelope()))
                    {
                        /*
                         * Close the captured producer immediately before the
                         * colocated CPU ticket transaction.  The following
                         * captured unit begins with authenticated ticket
                         * ingress, performs the ordered fold and shared branch,
                         * and continues through the next layer's GPU producer.
                         * Marking the post-ticket reducer instead would split
                         * that suffix a second time and manufacture two native
                         * executables for every one manual boundary.
                         *
                         * Every continuation participant owns this local
                         * expert terminal, so authority and LocalTP followers
                         * retain the same typed cutpoint without moving a
                         * collective across the CPU transaction.
                         */
                        graph.setHeterogeneousTicketUnitContract(
                            captured_local_expert_terminal,
                            GraphHeterogeneousTicketUnitContract{
                                .identity =
                                    prefix +
                                    "moe_overlay_pre_cpu_ticket_unit",
                            });

                        if (!rank_local_ticket_return_terminal.empty())
                        {
                            /*
                             * Declaratively place the root's host ticket unit
                             * after the same captured terminal used by every
                             * continuation participant.  This is submission
                             * order only; the ticket consumer observes its
                             * earlier exact publication edge and need not wait
                             * for unrelated work at this terminal.
                             */
                            graph.addDependency(
                                dispatch_name,
                                ordered_reduce_name);
                        }
                    }
                    if (sparse_graph_contract.ownsDispatchAuthority())
                    {
                        /*
                         * Portable remote packets follow submission of the
                         * complete outbound GPU wave without fencing it. A
                         * mapped retained parent instead launched every packet
                         * before this overlap body and makes its first ordered
                         * return fold consume the completed local reduction.
                         */
                        for (const auto &rank_batch_dispatch_name :
                             distributed_rank_batch_dispatch_nodes)
                        {
                            graph.addDependency(
                                rank_batch_dispatch_name,
                                ordered_reduce_name);
                        }
                    }

                    std::string publication_dependency =
                        ordered_reduce_name;
                    if (
                        sparse_graph_contract.ownsDispatchAuthority())
                    {
                        if (use_mapped_activation_parent &&
                            rank_local_ticket_return_terminal.empty())
                        {
                            /* Mapped GPU returns are already represented in
                             * canonical route slots and were consumed by the
                             * sole ordered reducer above. */
                        }
                        else if (has_rank_local_canonical_ticket_returns &&
                                 distributed_rank_batch_dispatch_nodes.empty())
                        {
                            /* Colocated CPU rows already occupy authenticated
                             * original slots in the canonical bank consumed by
                             * the reducer. No dense residual exists to merge. */
                        }
                        else if (use_mapped_activation_parent)
                        {
                            publication_dependency =
                                merge_direct_ticket_routes_after(
                                    ordered_reduce_name,
                                    /*annotate_inbound_capture_wave=*/false);
                        }
                        else
                        {
                        /*
                         * The final sparse ticket owns the aggregate remote
                         * routed contribution in ATTN_PROJ. Merge it into the
                         * root's canonically folded continuation-local result;
                         * the later shared-expert combine overwrites ATTN_PROJ
                         * only after this dependency completes.
                         */
                        ResidualAddStage::Params merge_params;
                        merge_params.device_id = device;
                        merge_params.input = buffers.attn_proj;
                        merge_params.residual = moe_output;
                        merge_params.output = moe_output;
                        merge_params.num_elements =
                            static_cast<size_t>(total_tokens) *
                            static_cast<size_t>(config_.d_model);
                        merge_params.input_buffer_id =
                            buffers.idFor(BufferId::ATTN_PROJ);
                        merge_params.residual_buffer_id =
                            buffers.idFor(BufferId::MOE_COMBINED_OUTPUT);
                        merge_params.output_buffer_id =
                            buffers.idFor(BufferId::MOE_COMBINED_OUTPUT);
                        const std::string merge_name =
                            prefix +
                            "moe_overlay_remote_routes_merge";
                        graph.addNode(
                            merge_name,
                            ComputeStageFactory::createResidualAdd(
                                merge_params),
                            device);
                        graph.addDependency(
                            merge_name,
                            ordered_reduce_name);
                        graph.addDependency(
                            merge_name,
                            last_return_reduce);
                        publication_dependency = merge_name;
                        }
                    }

                    const bool can_defer_combined_publication =
                        shared_expert_requires_tp_allreduce &&
                        has_shared_expert_branch &&
                        layer.shared_expert_gate_inp &&
                        planned_shared_device == device;
                    if (can_defer_combined_publication)
                    {
                        /* The shared partial is already graph-independent and
                         * ready while remote routed experts execute. Defer this
                         * intermediate publication so the next collective can
                         * reduce shared evidence directly to the fixed root;
                         * one later broadcast will publish the final combined
                         * row instead of transmitting two full-width tensors. */
                        deferred_overlay_combined_publication =
                            DeferredOverlayCombinedPublication{
                                .root_device_index =
                                    continuation_root_tp_index,
                                .participant_count = local_tp_ctx->degree(),
                                .routed_terminal = publication_dependency,
                                .capture_wave_identity =
                                    captured_overlay_inbound_wave,
                            };
                        if (!deferred_overlay_combined_publication->valid())
                        {
                            throw std::logic_error(
                                "Qwen35 MoE deferred overlay publication has an incomplete rooted topology contract");
                        }
                        ffn_terminal = publication_dependency;
                    }
                    else
                    {
                        TPLocalRootedCollectiveStage::Params broadcast_params;
                        broadcast_params.device_id = device;
                        broadcast_params.tp_ctx = local_tp_ctx;
                        broadcast_params.tensor = moe_output;
                        broadcast_params.count =
                            static_cast<size_t>(total_tokens) *
                            static_cast<size_t>(config_.d_model);
                        broadcast_params.dtype = CollectiveDataType::FLOAT32;
                        broadcast_params.operation =
                            TPLocalRootedCollectiveOperation::Broadcast;
                        broadcast_params.root_device_index =
                            continuation_root_tp_index;
                        broadcast_params.participant_device_index =
                            config_.tp_device_idx;
                        broadcast_params.stage_name =
                            prefix +
                            "moe_overlay_continuation_broadcast";
                        broadcast_params.tensor_buffer_id =
                            buffers.idFor(BufferId::MOE_COMBINED_OUTPUT);
                        if (shouldUseMoEOverlayMappedDensePublication(
                                route_transport,
                                broadcast_params.count * sizeof(float)))
                        {
                            broadcast_params
                                .mapped_dense_publication_exchange =
                                route_exchange;
                        }
                        const std::string publication_name =
                            broadcast_params.stage_name;
                        graph.addNode(
                            publication_name,
                            ComputeStageFactory::createTPLocalRootedCollective(
                                broadcast_params),
                            device);
                        graph.addDependency(
                            publication_name,
                            publication_dependency);
                        /*
                         * The root prepends return-ticket ingress and merge to
                         * this wave; the non-root begins at the broadcast. Both
                         * record the identical collective sequence.
                         */
                        graph.setGraphCaptureWaveContract(
                            publication_name,
                            GraphCaptureWaveContract{
                                .identity = captured_overlay_inbound_wave,
                            });
                        ffn_terminal = publication_name;
                    }
                }
                else if (captured_overlay_continuation)
                {
                    if (!sparse_graph_contract
                             .ownsDispatchAuthority() ||
                        captured_local_expert_terminal.empty())
                    {
                        throw std::runtime_error(
                            "Qwen35 MoE captured single-device continuation has no graph-local publication authority");
                    }

                    std::string ordered_reduce_dependency =
                        captured_local_expert_terminal;
                    if (!last_return_reduce.empty() &&
                        use_mapped_activation_parent)
                    {
                        ordered_reduce_dependency = last_return_reduce;
                    }
                    else if (!local_canonical_ticket_terminal.empty())
                    {
                        ordered_reduce_dependency =
                            local_canonical_ticket_terminal;
                    }

                    MoECanonicalRouteReduceStage::Params reduce_params;
                    reduce_params.device_id = device;
                    reduce_params.canonical_route_contributions =
                        canonical_route_contributions;
                    reduce_params.output = moe_output;
                    reduce_params.seq_len = total_tokens;
                    reduce_params.top_k = config_.moe.top_k;
                    reduce_params.d_model = config_.d_model;
                    reduce_params.canonical_route_arithmetic =
                        MoECanonicalRouteArithmeticPolicy::
                            PreweightedContributionThenOrderedAdd;
                    reduce_params.canonical_route_layout =
                        MoECanonicalRoutePublicationLayout::
                            DenseOriginalRouteSlots;
                    reduce_params.reduction_role =
                        MoECanonicalRouteReductionRole::RootOwner;
                    reduce_params.domain_route_assignment =
                        pinned_domain_route_assignment;
                    reduce_params.external_route_source =
                        use_mapped_activation_parent ||
                                has_rank_local_canonical_ticket_returns
                            ? MoEExternalCanonicalRouteSource::
                                  RootCanonicalRouteBank
                            : MoEExternalCanonicalRouteSource::
                                  DeferredDenseMerge;
                    reduce_params.runtime_route_weights =
                        pinned_runtime_route_weights;
                    reduce_params.overlay_route_placement =
                        pinned_route_placement;
                    reduce_params.canonical_route_contributions_buffer_id =
                        buffers.idFor(
                            BufferId::MOE_CANONICAL_ROUTE_CONTRIBUTIONS);
                    reduce_params.output_buffer_id =
                        buffers.idFor(BufferId::MOE_COMBINED_OUTPUT);

                    const std::string ordered_reduce_name =
                        prefix +
                        "moe_overlay_continuation_routes_ordered_reduce";
                    graph.addNode(
                        ordered_reduce_name,
                        ComputeStageFactory::createMoECanonicalRouteReduce(
                            reduce_params),
                        device);
                    graph.addDependency(
                        ordered_reduce_name,
                        ordered_reduce_dependency);

                    captured_overlay_routed_unit_terminal =
                        ordered_reduce_name;
                    if (requiresHeterogeneousTicketSegmentation(
                            graph.nativeCaptureEnvelope()))
                    {
                        /*
                         * The single-device continuation uses the same exact
                         * cutpoint as LocalTP: the local GPU expert terminal
                         * closes the producer unit immediately before the CPU
                         * ticket.  Ticket ingress and the ordered reduction
                         * belong to the following captured suffix; closing the
                         * unit on that reducer would add a redundant captured
                         * split after every CPU transaction.
                         */
                        graph.setHeterogeneousTicketUnitContract(
                            captured_local_expert_terminal,
                            GraphHeterogeneousTicketUnitContract{
                                .identity =
                                    prefix +
                                    "moe_overlay_pre_cpu_ticket_unit",
                            });

                        if (!rank_local_ticket_return_terminal.empty())
                        {
                            /*
                             * This is a submission-order edge, not a data
                             * dependency on the ticket publication itself.
                             * The host consumer reads the earlier authenticated
                             * publication record while the captured producer's
                             * remaining GPU work runs asynchronously.
                             */
                            graph.addDependency(
                                dispatch_name,
                                ordered_reduce_name);
                        }
                    }

                    std::string publication_dependency =
                        ordered_reduce_name;
                    if (use_mapped_activation_parent &&
                        !rank_local_ticket_return_terminal.empty())
                    {
                        publication_dependency =
                            merge_direct_ticket_routes_after(
                                ordered_reduce_name,
                                /*annotate_inbound_capture_wave=*/true);
                    }
                    else if (!last_return_reduce.empty() &&
                             !use_mapped_activation_parent &&
                             (!has_rank_local_canonical_ticket_returns ||
                              !distributed_rank_batch_dispatch_nodes.empty()))
                    {
                        /*
                         * A portable inter-node ticket arrives in ATTN_PROJ so
                         * it cannot overwrite the concurrently produced local
                         * contribution. Merge only after both asynchronous
                         * branches have published their exact completion edge.
                         */
                        ResidualAddStage::Params merge_params;
                        merge_params.device_id = device;
                        merge_params.input = buffers.attn_proj;
                        merge_params.residual = moe_output;
                        merge_params.output = moe_output;
                        merge_params.num_elements =
                            static_cast<size_t>(total_tokens) *
                            static_cast<size_t>(config_.d_model);
                        merge_params.input_buffer_id =
                            buffers.idFor(BufferId::ATTN_PROJ);
                        merge_params.residual_buffer_id =
                            buffers.idFor(BufferId::MOE_COMBINED_OUTPUT);
                        merge_params.output_buffer_id =
                            buffers.idFor(BufferId::MOE_COMBINED_OUTPUT);
                        const std::string merge_name =
                            prefix + "moe_overlay_remote_routes_merge";
                        graph.addNode(
                            merge_name,
                            ComputeStageFactory::createResidualAdd(
                                merge_params),
                            device);
                        graph.addDependency(
                            merge_name, captured_local_expert_terminal);
                        graph.addDependency(
                            merge_name, ordered_reduce_name);
                        graph.addDependency(
                            merge_name, last_return_reduce);
                        graph.setGraphCaptureWaveContract(
                            merge_name,
                            GraphCaptureWaveContract{
                                .identity = captured_overlay_inbound_wave,
                            });
                        publication_dependency = merge_name;
                    }
                    ffn_terminal = publication_dependency;
                }
                else if (distributed_dense_continuation)
                {
                    const auto &continuation_domain =
                        overlay_runtime_plan->continuationDomain();
                    if (!device.is_cpu() || !global_tp_ctx ||
                        global_tp_ctx->degree() <= 1 ||
                        global_tp_ctx->degree() !=
                            static_cast<int>(
                                continuation_domain.participants.size()) ||
                        global_tp_ctx->myIndex() != config_.tp_device_idx ||
                        continuation_root_participant < 0 ||
                        continuation_root_participant >=
                            global_tp_ctx->degree())
                    {
                        throw std::runtime_error(
                            "Qwen35 MoE distributed dense continuation requires "
                            "one aligned CPU GlobalTP participant per continuation "
                            "domain entry for layer " +
                            std::to_string(layer_idx));
                    }
                    if (sparse_graph_contract.ownsDispatchAuthority() &&
                        last_return_reduce.empty())
                    {
                        throw std::runtime_error(
                            "Qwen35 MoE distributed dense continuation source has no completed sparse return frontier");
                    }

                    /*
                     * A follower that owns no currently active expert may
                     * still be a dense TP participant. Its router node is the
                     * local readiness frontier; the rooted broadcast then
                     * waits for the source rank's complete sparse reduction.
                     * A follower with experts instead arrives here only after
                     * its authenticated return has been submitted.
                     */
                    const std::string publication_dependency =
                        last_return_reduce.empty()
                            ? prefix + "moe_routing"
                            : last_return_reduce;

                    MoECanonicalOutputBroadcastStage::Params
                        publication_params;
                    publication_params.device_id = device;
                    publication_params.tp_ctx = global_tp_ctx;
                    publication_params.output = moe_output;
                    publication_params.seq_len = total_tokens;
                    publication_params.d_model = config_.d_model;
                    publication_params.root_participant =
                        continuation_root_participant;
                    publication_params.stage_name =
                        prefix + "moe_overlay_continuation_broadcast";
                    publication_params.output_buffer_id =
                        buffers.idFor(BufferId::MOE_COMBINED_OUTPUT);

                    const std::string publication_name =
                        publication_params.stage_name;
                    graph.addNode(
                        publication_name,
                        ComputeStageFactory::
                            createMoECanonicalOutputBroadcast(
                                publication_params),
                        device);
                    graph.addDependency(
                        publication_name,
                        publication_dependency);
                    ffn_terminal = publication_name;
                }
                else
                {
                    ffn_terminal = last_return_reduce;
                }
            }
            else
            {
                auto expert_params = makeExpertParams(moe_output,
                                                      buffers.idFor(BufferId::MOE_COMBINED_OUTPUT),
                                                      {},
                                                      device);
                if (local_tp_ctx &&
                    total_tokens > 1 &&
                    prefill_routed_expert_assignment_policy == RoutedExpertAssignmentPolicy::LeastLoadedResident)
                {
                    expert_params.prefill_llep_tp_ctx = local_tp_ctx;
                    expert_params.prefill_llep_assignment_mode =
                        current_batch_llep_transfer_candidate
                            ? PrefillLLEPAssignmentMode::
                                  GraphPhasedCurrentBatch
                            : PrefillLLEPAssignmentMode::
                                  LogicalPositionResidentOnly;
                }
                if (!prepareExpertParams(expert_params, device))
                {
                    throw std::runtime_error(
                        "Qwen35 MoE graph failed to prepare expert parameters for layer " +
                        std::to_string(layer_idx) + " on " + device.to_string());
                }
                const bool routed_expert_output_is_partial =
                    expert_params.local_expert_count >= 0 ||
                    !expert_params.expert_mask.empty();
                const bool standard_cpu_canonical_publication =
                    device.is_cpu() &&
                    routed_expert_output_is_partial &&
                    needsMoEParticipantAllreduce();
                if (prefill_llep_transfer_candidate &&
                    (expert_params.use_runtime_row_grouping ||
                     prefix_runtime_device_rehydration))
                {
                    attachPrefillLLEPTransferBinding(
                        expert_params,
                        "standard routed expert LLEP grouped prefill");
                }

                const bool serial_decode_runtime_table_requested =
                    total_tokens == 1 &&
                    debugEnv().rocm.moe_grouped_decode &&
                    debugEnv().rocm.moe_device_routed_decode;
                const bool standard_gpu_runtime_table_requested =
                    device.is_gpu() &&
                    moe_runtime_table &&
                    (serial_decode_runtime_table_requested ||
                     grouped_main_verifier_layer);
                if (standard_gpu_runtime_table_requested)
                {
                    if (masked_local_tp_apportioned_decode_runtime_table)
                    {
                        const int participant_count =
                            local_tp_ctx ? local_tp_ctx->degree() : expert_params.participant_count;
                        const auto owner_participants =
                            routed_expert_ownership::ownerParticipantByExpert(
                                config_.moe.num_experts,
                                participant_count,
                                layer_idx,
                                config_.moe.owner_order);
                        if (!initializeMaskedLocalDecodeRuntimeTable(
                                moe_runtime_table,
                                layer_idx,
                                config_.moe.num_experts,
                                config_.moe.top_k,
                                config_.d_model,
                                expert_intermediate,
                                expert_params.expert_mask,
                                config_.tp_device_idx,
                                participant_count,
                                owner_participants,
                                {},
                                expert_params.prepared_gate_gemm,
                                expert_params.prepared_up_gemm,
                                expert_params.prepared_down_gemm,
                                device_state_publication_stream,
                                /*allow_existing_dynamic_bank=*/true,
                                "LocalTP expert-ID-apportioned masked GPU decode graph build"))
                        {
                            throw std::runtime_error(
                                "Qwen35 MoE graph failed to initialize masked LocalTP decode "
                                "runtime table for layer " +
                                std::to_string(layer_idx) + " on " + device.to_string());
                        }
                    }
                    else
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
                                FullLocalDecodeRuntimePolicy::SingleParticipant,
                                /*local_participant=*/0,
                                /*participant_count=*/1,
                                /*owner_participants=*/{},
                                expert_params.prepared_gate_gemm,
                                expert_params.prepared_up_gemm,
                                expert_params.prepared_down_gemm,
                                device_state_publication_stream,
                                "single-device GPU decode graph build"))
                        {
                            throw std::runtime_error(
                                "Qwen35 MoE graph failed to initialize device decode "
                                "runtime table for layer " +
                                std::to_string(layer_idx) + " on " + device.to_string());
                        }
                    }

                    /*
                     * A grouped main verifier must retain its final device route
                     * assignment until accepted-state publication commits the
                     * accepted prefix.  The same runtime table therefore owns
                     * both grouping scratch and the deferred route ledger.  This
                     * is not optional telemetry: omitting the binding lets the
                     * verifier compute rows but makes their state transaction
                     * impossible to commit.
                     */
                    if (grouped_main_verifier_layer)
                        expert_params.use_runtime_row_grouping = true;
                }

                if (standard_cpu_canonical_publication ||
                    route_accumulation_policy ==
                        MoERouteAccumulationPolicy::
                            IndependentRouteSlotsThenOrderedFold)
                {
                    if (!canonical_route_contributions)
                    {
                        throw std::runtime_error(
                            "Qwen35 MoE canonical route-slot publication "
                            "requires canonical device storage for layer " +
                            std::to_string(layer_idx) + " on " +
                            device.to_string());
                    }
                    expert_params.canonical_route_contributions =
                        canonical_route_contributions;
                    expert_params.canonical_route_arithmetic =
                        standard_cpu_canonical_publication
                            ? MoECanonicalRouteArithmeticPolicy::
                                  UnweightedExpertRowThenOrderedFMA
                            : MoECanonicalRouteArithmeticPolicy::
                                  PreweightedContributionThenOrderedAdd;
                    expert_params.canonical_route_layout =
                        standard_cpu_canonical_publication
                            ? MoECanonicalRoutePublicationLayout::
                                  PackedIndexedRouteRows
                            : MoECanonicalRoutePublicationLayout::
                                  DenseOriginalRouteSlots;
                    expert_params.canonical_route_contributions_buffer_id =
                        buffers.idFor(
                            BufferId::MOE_CANONICAL_ROUTE_CONTRIBUTIONS);
                }

                auto expert_stage_owner =
                    ComputeStageFactory::createMoEExpertCompute(expert_params);
                graph.addNode(prefix + "moe_expert_ffn",
                              std::move(expert_stage_owner), device);
                const std::string rebalance_apply_dependency =
                    maybeInsertGraphSideRebalance(
                        "standard routed expert path",
                        prefix + "moe_expert_ffn");
                graph.addDependency(
                    prefix + "moe_expert_ffn",
                    rebalance_apply_dependency.empty()
                        ? prefix + "moe_routing"
                        : rebalance_apply_dependency);
                ffn_terminal = prefix + "moe_expert_ffn";

                if (standard_cpu_canonical_publication)
                {
                    /*
                     * CPU Dynamic ownership changes which participant computes
                     * each route, but it must not change either arithmetic or
                     * communication volume by `top_k`. Each participant has
                     * already emitted only its live raw route rows as indexed
                     * packed records. Gather those records to rank zero, replay
                     * the original increasing-slot weighted FMA exactly once,
                     * then broadcast only the compact final row.
                     */
                    auto rebalance_sidebands =
                        takeGraphRebalanceSidebandsForAllreduce();
                    if (!rebalance_sidebands.empty() ||
                        !graph_rebalance_collect_node.empty() ||
                        graph_rebalance_plan_after_sideband_params.has_value() ||
                        graph_rebalance_pack_payload_params.has_value() ||
                        graph_rebalance_unpack_payload_params.has_value())
                    {
                        throw std::logic_error(
                            "Qwen35 MoE CPU packed route publication cannot "
                            "consume GPU graph-side rebalance sidebands for layer " +
                            std::to_string(layer_idx));
                    }

                    constexpr int canonical_root_participant = 0;
                    MoECanonicalRouteGatherStage::Params gather_params;
                    gather_params.device_id = device;
                    gather_params.tp_ctx = config_.tp_ctx;
                    gather_params.packed_route_records =
                        canonical_route_contributions;
                    gather_params.seq_len = total_tokens;
                    gather_params.top_k = config_.moe.top_k;
                    gather_params.d_model = config_.d_model;
                    gather_params.root_participant =
                        canonical_root_participant;
                    gather_params.stage_name =
                        prefix + "moe_canonical_routes_gather_to_root";
                    gather_params.packed_route_records_buffer_id =
                        buffers.idFor(
                            BufferId::MOE_CANONICAL_ROUTE_CONTRIBUTIONS);
                    const std::string gather_name = gather_params.stage_name;
                    graph.addNode(
                        gather_name,
                        ComputeStageFactory::createMoECanonicalRouteGather(
                            gather_params),
                        device);
                    graph.addDependency(
                        gather_name,
                        prefix + "moe_expert_ffn");

                    MoECanonicalRouteReduceStage::Params reduce_params;
                    reduce_params.device_id = device;
                    reduce_params.canonical_route_contributions =
                        canonical_route_contributions;
                    reduce_params.routing_weights = routing_weights;
                    reduce_params.output = moe_output;
                    reduce_params.seq_len = total_tokens;
                    reduce_params.top_k = config_.moe.top_k;
                    reduce_params.d_model = config_.d_model;
                    reduce_params.canonical_route_arithmetic =
                        MoECanonicalRouteArithmeticPolicy::
                            UnweightedExpertRowThenOrderedFMA;
                    reduce_params.canonical_route_layout =
                        MoECanonicalRoutePublicationLayout::
                            PackedIndexedRouteRows;
                    reduce_params.reduction_role =
                        config_.tp_device_idx ==
                                canonical_root_participant
                            ? MoECanonicalRouteReductionRole::RootOwner
                            : MoECanonicalRouteReductionRole::
                                  NonRootParticipant;
                    reduce_params.canonical_route_contributions_buffer_id =
                        buffers.idFor(
                            BufferId::MOE_CANONICAL_ROUTE_CONTRIBUTIONS);
                    reduce_params.routing_weights_buffer_id =
                        buffers.idFor(BufferId::MOE_EXPERT_WEIGHTS);
                    reduce_params.output_buffer_id =
                        buffers.idFor(BufferId::MOE_COMBINED_OUTPUT);

                    const std::string reduce_name =
                        prefix + "moe_canonical_routes_ordered_fma";
                    graph.addNode(
                        reduce_name,
                        ComputeStageFactory::createMoECanonicalRouteReduce(
                            reduce_params),
                        device);
                    graph.addDependency(reduce_name, gather_name);

                    MoECanonicalOutputBroadcastStage::Params broadcast_params;
                    broadcast_params.device_id = device;
                    broadcast_params.tp_ctx = config_.tp_ctx;
                    broadcast_params.output = moe_output;
                    broadcast_params.seq_len = total_tokens;
                    broadcast_params.d_model = config_.d_model;
                    broadcast_params.root_participant =
                        canonical_root_participant;
                    broadcast_params.stage_name =
                        prefix + "moe_canonical_routes_broadcast";
                    broadcast_params.output_buffer_id =
                        buffers.idFor(BufferId::MOE_COMBINED_OUTPUT);
                    const std::string broadcast_name =
                        broadcast_params.stage_name;
                    graph.addNode(
                        broadcast_name,
                        ComputeStageFactory::
                            createMoECanonicalOutputBroadcast(
                                broadcast_params),
                        device);
                    graph.addDependency(broadcast_name, reduce_name);
                    ffn_terminal = broadcast_name;
                }
                else if (route_accumulation_policy ==
                    MoERouteAccumulationPolicy::
                        IndependentRouteSlotsThenOrderedFold)
                {
                    MoECanonicalRouteReduceStage::Params reduce_params;
                    reduce_params.device_id = device;
                    reduce_params.canonical_route_contributions =
                        canonical_route_contributions;
                    reduce_params.output = moe_output;
                    reduce_params.seq_len = total_tokens;
                    reduce_params.top_k = config_.moe.top_k;
                    reduce_params.d_model = config_.d_model;
                    reduce_params.canonical_route_arithmetic =
                        MoECanonicalRouteArithmeticPolicy::
                            PreweightedContributionThenOrderedAdd;
                    reduce_params.canonical_route_layout =
                        MoECanonicalRoutePublicationLayout::
                            DenseOriginalRouteSlots;
                    reduce_params.reduction_role =
                        MoECanonicalRouteReductionRole::RootOwner;
                    reduce_params.canonical_route_contributions_buffer_id =
                        buffers.idFor(
                            BufferId::MOE_CANONICAL_ROUTE_CONTRIBUTIONS);
                    reduce_params.output_buffer_id =
                        buffers.idFor(BufferId::MOE_COMBINED_OUTPUT);

                    const std::string reduce_name =
                        prefix + "moe_local_canonical_routes_reduce";
                    graph.addNode(
                        reduce_name,
                        ComputeStageFactory::createMoECanonicalRouteReduce(
                            reduce_params),
                        device);
                    graph.addDependency(
                        reduce_name,
                        prefix + "moe_expert_ffn");
                    ffn_terminal = reduce_name;
                }

                LOG_TRACE(
                    "[Qwen35MoEGraph] Layer " << layer_idx
                                              << " route_accumulation="
                                              << moeRouteAccumulationPolicyToString(
                                                     route_accumulation_policy)
                                              << " device="
                                              << device.to_string());

                // Qwen35 MoE expert weights are normally replicated, so every rank
                // computes the full routed-expert contribution. GPU and legacy
                // non-participant partial paths retain their compact collective;
                // CPU participant ownership was lowered canonically above.
                if (!standard_cpu_canonical_publication &&
                    routed_expert_output_is_partial &&
                    needsTPAllreduce())
                {
                    size_t allreduce_count = static_cast<size_t>(total_tokens) * static_cast<size_t>(config_.d_model);
                    std::string ar_name = prefix + "moe_expert_allreduce";
                    auto rebalance_sidebands = takeGraphRebalanceSidebandsForAllreduce();
                    auto allreduce_stage = createTPAllreduceStage(
                        moe_output, allreduce_count, device, layer_idx,
                        /*is_attention=*/false, ar_name, buffers.idFor(BufferId::MOE_COMBINED_OUTPUT),
                        std::move(rebalance_sidebands));
                    if (allreduce_stage)
                    {
                        graph.addNode(ar_name, std::move(allreduce_stage), device);
                        graph.addDependency(ar_name, prefix + "moe_expert_ffn");
                        ffn_terminal = ar_name;
                        maybeAddGraphRebalancePayloadStageAfterSideband(
                            ar_name,
                            ffn_terminal);
                    }
                }

            }
        }

        // =====================================================================
        // Stage 4: Shared Expert FFN (always-active dense SwiGLU)
        // =====================================================================

        if (!moe_combined_output_ready &&
            layer.shared_expert_gate && layer.shared_expert_up && layer.shared_expert_down && shared_output)
        {
            DeviceId shared_device = planned_shared_device;
            /*
             * Shared experts are always-on dense FFNs even though their GGUF
             * names contain "expert". Phase-split decode therefore binds the
             * complete replicated shared-expert weights on every participant,
             * exactly like replicated attention and dense FFN weights. Only
             * the tensor-parallel prefill view produces a partial down row that
             * requires a collective. Keeping this policy in one named value
             * prevents the gate/combine lowering and the collective lowering
             * from disagreeing about whether the branch is already complete.
             */
            if (overlay_runtime_plan)
            {
                const auto &continuation_domain = overlay_runtime_plan->continuationDomain();
                const auto &shared_domain = overlay_runtime_plan->sharedExpertDomain();

                LOG_TRACE("[Qwen35MoEGraph] Layer " << layer_idx
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
            shared_params.gate_scratch =
                buffers.get(buffers.idFor(BufferId::MOE_GATE_SCRATCH));
            shared_params.up_scratch =
                buffers.get(buffers.idFor(BufferId::MOE_UP_SCRATCH));
            shared_params.seq_len = total_tokens;
            shared_params.d_model = config_.d_model;
            shared_params.intermediate = shared_intermediate;
            shared_params.input_buffer_id = buffers.idFor(BufferId::NORMALIZED);
            shared_params.output_buffer_id = buffers.idFor(BufferId::MOE_SHARED_EXPERT_OUTPUT);
            shared_params.gate_scratch_buffer_id =
                buffers.idFor(BufferId::MOE_GATE_SCRATCH);
            shared_params.up_scratch_buffer_id =
                buffers.idFor(BufferId::MOE_UP_SCRATCH);
            /*
             * Shared-expert verifier rows must match ordinary serial decode,
             * not a separate dense GEMM oracle.  GPU serial decode uses the MoE
             * grouped table-decode helpers for the always-active shared expert,
             * so verifier-sized GPU batches use the sibling grouped table-prefill
             * route. This keeps runtime-M verifier execution economical while preserving the exact
             * descriptor family, split-K layout, and reduction order exercised
             * by production M=1 decode.
             */
            const bool shared_grouped_verifier_prefill =
                forceGroupedSharedMoEVerifierPrefill(shared_device);
            const bool shared_gpu_table_verifier_prefill =
                shared_grouped_verifier_prefill &&
                (shared_device.is_cuda() || shared_device.is_rocm());
            shared_params.force_grouped_verifier_prefill_for_decode =
                shared_gpu_table_verifier_prefill;
            /*
             * Router-row reuse is an optimization, not a prerequisite for the
             * grouped shared-expert verifier.  Deterministic execution disables
             * that reuse on both GPU backends so serial decode and grouped
             * verification quantize their inputs independently in the same
             * arithmetic regime.  Bind a required publication only when the
             * selected backend will actually produce it; otherwise the grouped
             * pipeline owns its normal graph-local quantization buffers.
             */
            const bool shared_router_q8_reuse_enabled =
                (shared_device.is_cuda() &&
                 debugEnv().gemm.cuda_moe_reuse_router_q8_hidden) ||
                (shared_device.is_rocm() &&
                 debugEnv().rocm.moe_reuse_router_q8_hidden);
            const bool shared_requires_router_q8_publication =
                shared_gpu_table_verifier_prefill &&
                shared_router_q8_reuse_enabled;
            if (shared_requires_router_q8_publication)
            {
                shared_params.required_router_q8_publication =
                    routed_pipeline_kernel_owner->router_q8_publication;
            }
            shared_params.force_decode_equivalent_verifier_prefill =
                (!shared_gpu_table_verifier_prefill &&
                 (!shared_grouped_verifier_prefill &&
                  forceDecodeEquivalentMoEVerifier(shared_device)));
            /*
             * Phase-split MTP sidecars execute against the replicated dense-decode
             * weight view rather than the ordinary per-rank TP decode view.  The
             * normal GPU grouped shared-expert decode shortcut is tuned for that
             * ordinary one-token decode path and owns a separate pointer-table
             * readiness contract.  MTP sidecar rows are verifier rows: they may
             * become live state, so their one-row shared-expert math must flow
             * through the decode-equivalent verifier path unless a grouped
             * shortcut has passed the same strict serial-decode proof.
             */
            shared_params.disable_grouped_decode_shortcut =
                mtp_sidecar_context &&
                (config_.dense_tp_decode_replicated ||
                 config_.mtpUsesReplicatedDenseSidecarBinding());
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
            if (shared_requires_router_q8_publication)
            {
                /*
                 * The GPU router quantizes the normalized verifier rows once and
                 * publishes those device-resident Q8 rows for both the routed and
                 * shared grouped pipelines.  The shared stage therefore consumes
                 * router-owned state even though its visible tensor input is still
                 * NORMALIZED.  Encode that hidden producer/consumer relationship in
                 * the graph: relying on the incidental insertion order of sibling
                 * nodes can let the shared stage reuse the previous layer's Q8
                 * publication when a fresh graph receives a different topological
                 * ordering. This edge applies equally to main-verifier and MTP
                 * sidecar graphs when reuse is enabled, and remains valid during
                 * whole-graph capture. Standalone quantization has no hidden
                 * producer and deliberately keeps the two branches parallel.
                 */
                graph.addDependency(
                    prefix + "shared_expert_ffn",
                    prefix + "moe_routing");
            }
            const bool main_verifier_rows =
                !mtp_sidecar_context &&
                config_.compute_all_position_logits &&
                total_tokens >= 1;
            /*
             * The accepted shared-verifier routes own their branch-local math:
             * CUDA and ROCm use the grouped table-prefill verifier route. Any route
             * outside those explicit grouped implementations keeps the
             * conservative dependency until it has its own branch-scoped
             * workspace proof.
             */
            const bool shared_verifier_owns_branch_local_math =
                main_verifier_rows &&
                (shared_gpu_table_verifier_prefill ||
                 shared_params.force_decode_equivalent_verifier_prefill ||
                 canonical_publication_lowering.usesRankBanks());
            if (main_verifier_rows &&
                !shared_verifier_owns_branch_local_math &&
                !ffn_terminal.empty())
            {
                /*
                 * A branch whose implementation shares mutable backend bridge
                 * state with routed execution is serialized by policy. The
                 * grouped GPU and canonical rank-bank implementations own
                 * distinct persistent workspaces and therefore remain parallel.
                 */
                graph.addDependency(prefix + "shared_expert_ffn", ffn_terminal);
            }
            shared_ffn_last = prefix + "shared_expert_ffn";

            if (canonical_publication_lowering.usesRankBanks())
            {
                if (!canonical_route_contributions || !buffers.normalized ||
                    !buffers.attn_proj || !moe_output ||
                    !layer.shared_expert_gate_inp || !local_tp_ctx ||
                    canonical_publication_lowering.root_participant < 0 ||
                    canonical_publication_lowering.participant_count <= 0 ||
                    canonical_publication_lowering.routed_producer.empty() ||
                    canonical_publication_lowering.rooted_reduce_node.empty() ||
                    canonical_publication_lowering.post_collective_terminal.empty())
                {
                    throw std::logic_error(
                        "Qwen35 MoE canonical rank-bank publication has an "
                        "incomplete graph contract for layer " +
                        std::to_string(layer_idx) + " on " +
                        device.to_string());
                }

                MoESharedExpertRankBankPublishStage::Params publish_params;
                publish_params.device_id = device;
                publish_params.shared_output = shared_output;
                publish_params.canonical_publication =
                    canonical_route_contributions;
                publish_params.seq_len = total_tokens;
                publish_params.top_k = config_.moe.top_k;
                publish_params.d_model = config_.d_model;
                publish_params.participant_device_index =
                    config_.tp_device_idx;
                publish_params.participant_count =
                    canonical_publication_lowering.participant_count;
                publish_params.active_row_count_device =
                    device.is_gpu() && batch_size == 1
                        ? sequence_lengths_device
                        : nullptr;
                publish_params.shared_output_buffer_id =
                    buffers.idFor(BufferId::MOE_SHARED_EXPERT_OUTPUT);
                publish_params.canonical_publication_buffer_id =
                    buffers.idFor(
                        BufferId::MOE_CANONICAL_ROUTE_CONTRIBUTIONS);

                const std::string publish_name =
                    prefix + "moe_shared_rank_bank_publish";
                graph.addNode(
                    publish_name,
                    ComputeStageFactory::
                        createMoESharedExpertRankBankPublish(
                            publish_params),
                    device);
                graph.addDependency(
                    publish_name,
                    prefix + "shared_expert_ffn");
                graph.addDependency(
                    publish_name,
                    canonical_publication_lowering.routed_producer);
                graph.addDependency(
                    canonical_publication_lowering.rooted_reduce_node,
                    publish_name);

                MoECanonicalPublicationFinalizeStage::Params finalize_params;
                finalize_params.device_id = device;
                finalize_params.input = buffers.normalized;
                finalize_params.gate_inp = layer.shared_expert_gate_inp;
                finalize_params.canonical_publication =
                    canonical_route_contributions;
                finalize_params.routed_output = moe_output;
                finalize_params.shared_output = shared_output;
                finalize_params.combined_output = buffers.attn_proj;
                finalize_params.seq_len = total_tokens;
                finalize_params.top_k = config_.moe.top_k;
                finalize_params.d_model = config_.d_model;
                finalize_params.participant_device_index =
                    config_.tp_device_idx;
                finalize_params.root_device_index =
                    canonical_publication_lowering.root_participant;
                finalize_params.participant_count =
                    canonical_publication_lowering.participant_count;
                finalize_params.active_row_count_device =
                    device.is_gpu() && batch_size == 1
                        ? sequence_lengths_device
                        : nullptr;
                finalize_params.input_buffer_id =
                    buffers.idFor(BufferId::NORMALIZED);
                finalize_params.canonical_publication_buffer_id =
                    buffers.idFor(
                        BufferId::MOE_CANONICAL_ROUTE_CONTRIBUTIONS);
                finalize_params.routed_output_buffer_id =
                    buffers.idFor(BufferId::MOE_COMBINED_OUTPUT);
                finalize_params.shared_output_buffer_id =
                    buffers.idFor(BufferId::MOE_SHARED_EXPERT_OUTPUT);
                finalize_params.combined_output_buffer_id =
                    buffers.idFor(BufferId::ATTN_PROJ);

                const std::string finalize_name =
                    prefix + "moe_canonical_publication_finalize";
                graph.addNode(
                    finalize_name,
                    ComputeStageFactory::
                        createMoECanonicalPublicationFinalize(
                            finalize_params),
                    device);
                graph.addDependency(
                    finalize_name,
                    canonical_publication_lowering.post_collective_terminal);

                TPLocalRootedCollectiveStage::Params broadcast_params;
                broadcast_params.device_id = device;
                broadcast_params.tp_ctx = local_tp_ctx;
                broadcast_params.tensor = buffers.attn_proj;
                broadcast_params.count =
                    static_cast<size_t>(total_tokens) *
                    static_cast<size_t>(config_.d_model);
                broadcast_params.dtype = CollectiveDataType::FLOAT32;
                broadcast_params.operation =
                    TPLocalRootedCollectiveOperation::Broadcast;
                broadcast_params.root_device_index =
                    canonical_publication_lowering.root_participant;
                broadcast_params.participant_device_index =
                    config_.tp_device_idx;
                broadcast_params.stage_name =
                    prefix + "moe_canonical_publication_broadcast";
                broadcast_params.tensor_buffer_id =
                    buffers.idFor(BufferId::ATTN_PROJ);

                const std::string broadcast_name =
                    broadcast_params.stage_name;
                graph.addNode(
                    broadcast_name,
                    ComputeStageFactory::createTPLocalRootedCollective(
                        broadcast_params),
                    device);
                graph.addDependency(broadcast_name, finalize_name);

                moe_combined_output_ready = true;
                shared_ffn_last = broadcast_name;
                ffn_terminal = broadcast_name;

                LOG_TRACE(
                    "[Qwen35MoEGraph] Layer "
                    << layer_idx << " lowered "
                    << moeParticipantPublicationPolicyToString(
                           canonical_publication_lowering.policy)
                    << " participants="
                    << canonical_publication_lowering.participant_count
                    << " root="
                    << canonical_publication_lowering.root_participant
                    << " device=" << device.to_string());
            }

            /*
             * Input-parallel prefill shared-expert down rows are reduced before
             * the replicated sigmoid gate, preserving the serial LocalTP branch
             * arithmetic independently of routed-expert placement. Replicated
             * decode rows are already complete and must never be summed again.
             */
            if (shared_expert_requires_tp_allreduce &&
                !canonical_publication_lowering.usesRankBanks())
            {
                const size_t allreduce_count =
                    static_cast<size_t>(total_tokens) *
                    static_cast<size_t>(config_.d_model);
                const bool rooted_overlay_shared_reduction =
                    deferred_overlay_combined_publication.has_value();
                std::string ar_name =
                    prefix +
                    (rooted_overlay_shared_reduction
                         ? "shared_expert_reduce_to_overlay_root"
                         : "shared_expert_allreduce");
                std::vector<TPAllreduceSidebandWorkspaceBinding> rebalance_sidebands;
                if (shared_device == device)
                {
                    rebalance_sidebands = takeGraphRebalanceSidebandsForAllreduce();
                    appendRebalanceSidebands(
                        rebalance_sidebands,
                        takeCurrentBatchLLEPPayloadSideband());
                }
                std::unique_ptr<IComputeStage> allreduce_stage;
                if (rooted_overlay_shared_reduction)
                {
                    const auto &publication =
                        *deferred_overlay_combined_publication;
                    if (!publication.valid() || !local_tp_ctx ||
                        config_.tp_device_idx < 0 ||
                        config_.tp_device_idx >= publication.participant_count ||
                        local_tp_ctx->degree() != publication.participant_count)
                    {
                        throw std::logic_error(
                            "Qwen35 MoE rooted overlay shared reduction has an incomplete participant contract");
                    }
                    TPLocalRootedCollectiveStage::Params reduce_params;
                    reduce_params.device_id = shared_device;
                    reduce_params.tp_ctx = local_tp_ctx;
                    reduce_params.tensor = shared_output;
                    reduce_params.count = allreduce_count;
                    reduce_params.dtype = CollectiveDataType::FLOAT32;
                    reduce_params.operation =
                        TPLocalRootedCollectiveOperation::ReduceSum;
                    reduce_params.root_device_index =
                        publication.root_device_index;
                    reduce_params.participant_device_index =
                        config_.tp_device_idx;
                    reduce_params.stage_name = ar_name;
                    reduce_params.tensor_buffer_id =
                        buffers.idFor(BufferId::MOE_SHARED_EXPERT_OUTPUT);
                    reduce_params.sideband_workspace_bindings =
                        std::move(rebalance_sidebands);
                    allreduce_stage =
                        ComputeStageFactory::createTPLocalRootedCollective(
                            reduce_params);
                }
                else
                {
                    allreduce_stage = createTPAllreduceStage(
                        shared_output,
                        allreduce_count,
                        shared_device,
                        layer_idx,
                        /*is_attention=*/false,
                        ar_name,
                        buffers.idFor(BufferId::MOE_SHARED_EXPERT_OUTPUT),
                        std::move(rebalance_sidebands));
                }
                if (allreduce_stage)
                {
                    graph.addNode(ar_name, std::move(allreduce_stage), shared_device);
                    graph.addDependency(ar_name, prefix + "shared_expert_ffn");
                    if (rooted_overlay_shared_reduction &&
                        !captured_overlay_routed_unit_terminal.empty())
                    {
                        /*
                         * The shared partial is ready before the heterogeneous
                         * ticket, but its rooted collective belongs to the
                         * post-ticket publication unit.  Making the collective
                         * consume the routed-unit terminal gives every LocalTP
                         * participant the same collective sequence in the same
                         * capture wave.  The reverse edge lets the authority
                         * enqueue an unmatched RCCL reduction before its manual
                         * CPU transaction while the peer waits at the next
                         * capture-wave rendezvous.  On devices whose collective
                         * wait kernel occupies the compute units, that cycle can
                         * also starve the ticket publication which would release
                         * the manual transaction.
                         */
                        graph.addDependency(
                            ar_name,
                            captured_overlay_routed_unit_terminal);
                    }
                    if (current_batch_llep_plan_params.has_value())
                    {
                        if (current_batch_llep_plan_node.empty() ||
                            current_batch_llep_apply_node.empty() ||
                            current_batch_llep_expert_node.empty() ||
                            !current_batch_llep_apply_params.has_value() ||
                            !current_batch_llep_payload_sideband_taken)
                        {
                            throw std::logic_error(
                                "Qwen35 MoE current-batch LLEP shared-collective binding is incomplete for layer " +
                                std::to_string(layer_idx));
                        }
                        graph.addDependency(
                            ar_name,
                            current_batch_llep_plan_node);
                        graph.addNode(
                            current_batch_llep_apply_node,
                            ComputeStageFactory::createMoEGPUCurrentBatchLLEP(
                                *current_batch_llep_apply_params),
                            device);
                        graph.addDependency(
                            current_batch_llep_apply_node,
                            ar_name);
                        graph.addDependency(
                            current_batch_llep_expert_node,
                            current_batch_llep_apply_node);
                    }
                    if (!rooted_overlay_shared_reduction &&
                        routed_overlay_has_distributed_sparse_protocol &&
                        !ffn_terminal.empty())
                    {
                        /*
                         * Both rank graphs must finish the same dispatch,
                         * participant compute, and return sequence before one
                         * of them enters this independent dense collective.
                         * The edge orders protocols without preventing the
                         * shared-expert GEMM from running while routed work is
                         * in flight.
                         */
                        graph.addDependency(ar_name, ffn_terminal);
                    }
                    shared_ffn_last = ar_name;
                    if (shared_device == device)
                    {
                        maybeAddGraphRebalancePayloadStageAfterSideband(
                            ar_name,
                            shared_ffn_last);
                    }
                }
            }

            // Stage 4b: Sigmoid gate on shared expert output
            if (layer.shared_expert_gate_inp &&
                !canonical_publication_lowering.usesRankBanks())
            {
                const bool rooted_overlay_finalize =
                    deferred_overlay_combined_publication.has_value();
                const bool can_fuse_gate_and_combine =
                    (!shared_expert_requires_tp_allreduce ||
                     rooted_overlay_finalize) &&
                    shared_device == device &&
                    moe_output && buffers.attn_proj;
                const bool gate_writes_combined_output =
                    can_fuse_gate_and_combine;

                SharedExpertGateStage::Params gate_params;
                gate_params.device_id = shared_device;
                gate_params.input = buffers.normalized;
                gate_params.gate_inp = layer.shared_expert_gate_inp;
                gate_params.shared_output = shared_output;
                gate_params.seq_len = total_tokens;
                gate_params.d_model = config_.d_model;
                gate_params.active_row_count_device =
                    shared_device.is_gpu() && batch_size == 1
                        ? sequence_lengths_device
                        : nullptr;
                if (rooted_overlay_finalize)
                {
                    const auto &publication =
                        *deferred_overlay_combined_publication;
                    gate_params.execution_role =
                        config_.tp_device_idx ==
                                publication.root_device_index
                            ? SharedExpertGateStage::ExecutionRole::RootOwner
                            : SharedExpertGateStage::ExecutionRole::
                                  NonRootObserver;
                }
                gate_params.input_buffer_id = buffers.idFor(BufferId::NORMALIZED);
                gate_params.output_buffer_id = buffers.idFor(BufferId::MOE_SHARED_EXPERT_OUTPUT);
                if (gate_writes_combined_output)
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
                if (gate_writes_combined_output)
                {
                    /* The root consumes both rooted shared bytes and its
                     * complete routed row. The non-root node is a typed no-op,
                     * but keeps the identical dependency and collective order
                     * before final publication. */
                    graph.addDependency(
                        prefix + "shared_expert_gate",
                        rooted_overlay_finalize
                            ? deferred_overlay_combined_publication
                                  ->routed_terminal
                            : ffn_terminal);
                    moe_combined_output_ready = true;
                }
                shared_ffn_last = prefix + "shared_expert_gate";
                if (moe_combined_output_ready)
                {
                    ffn_terminal = prefix + "shared_expert_gate";
                }

                if (rooted_overlay_finalize)
                {
                    const auto &publication =
                        *deferred_overlay_combined_publication;
                    if (!gate_writes_combined_output || !publication.valid() ||
                        !local_tp_ctx)
                    {
                        throw std::logic_error(
                            "Qwen35 MoE deferred overlay publication did not produce a root-owned combined row");
                    }

                    TPLocalRootedCollectiveStage::Params broadcast_params;
                    broadcast_params.device_id = device;
                    broadcast_params.tp_ctx = local_tp_ctx;
                    broadcast_params.tensor = buffers.attn_proj;
                    broadcast_params.count =
                        static_cast<size_t>(total_tokens) *
                        static_cast<size_t>(config_.d_model);
                    broadcast_params.dtype = CollectiveDataType::FLOAT32;
                    broadcast_params.operation =
                        TPLocalRootedCollectiveOperation::Broadcast;
                    broadcast_params.root_device_index =
                        publication.root_device_index;
                    broadcast_params.participant_device_index =
                        config_.tp_device_idx;
                    broadcast_params.stage_name =
                        prefix + "moe_overlay_combined_broadcast";
                    broadcast_params.tensor_buffer_id =
                        buffers.idFor(BufferId::ATTN_PROJ);
                    if (shouldUseMoEOverlayMappedDensePublication(
                            config_.moe.node_local_route_transport,
                            broadcast_params.count * sizeof(float)))
                    {
                        broadcast_params
                            .mapped_dense_publication_exchange =
                            config_.moe.node_local_route_exchange;
                    }
                    const std::string broadcast_name =
                        broadcast_params.stage_name;
                    graph.addNode(
                        broadcast_name,
                        ComputeStageFactory::createTPLocalRootedCollective(
                            broadcast_params),
                        device);
                    graph.addDependency(
                        broadcast_name, prefix + "shared_expert_gate");
                    graph.setGraphCaptureWaveContract(
                        broadcast_name,
                        GraphCaptureWaveContract{
                            .identity = publication.capture_wave_identity,
                        });
                    shared_ffn_last = broadcast_name;
                    ffn_terminal = broadcast_name;
                }
            }
        }

        // =====================================================================
        // Stage 5: Combine expert output + shared expert output → attn_proj
        // =====================================================================
        // The combined MoE output goes to attn_proj so that the next layer's
        // FusedResidualNormStage handles the residual add automatically.
        {
            if (moe_combined_output_ready)
            {
                // A fused shared gate or canonical rooted finalizer already
                // wrote ATTN_PROJ; no standalone residual add is permitted.
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

        if (last_device_rebalance_decode_layer &&
            !env.moe_rebalance.device_rebalance_maintenance_graph &&
            !ffn_terminal.empty())
        {
            const auto binding_it =
                moe_graph_rebalance_bindings_.find(graphRebalanceDomainKey());
            if (binding_it != moe_graph_rebalance_bindings_.end())
            {
                if (deviceMoERebalanceModeUsesTransferSlots(binding_it->second.transfer_mode) &&
                    binding_it->second.local_transfer_slot_count > 0)
                {
                    const auto transfer_directory_it =
                        moe_transfer_slot_directories_.find(binding_it->second.transfer_key);
                    const auto transfer_state_it =
                        moe_rebalance_transfer_states_.find(
                            binding_it->second.transfer_key + ":workspace=" +
                            binding_it->second.workspace_name);
                    if (transfer_directory_it == moe_transfer_slot_directories_.end() ||
                        !transfer_directory_it->second ||
                        transfer_state_it == moe_rebalance_transfer_states_.end() ||
                        !transfer_state_it->second)
                    {
                        throw std::runtime_error(
                            "Qwen35 MoE graph-side rebalance lost its late transfer-join binding for layer " +
                            std::to_string(layer_idx) + " on " + device.to_string());
                    }

                    const std::string join_node =
                        prefix + "moe_device_rebalance_transfer_join";
                    MoEDeviceRebalanceStage::Params join_params;
                    join_params.device_id = device;
                    join_params.tp_ctx = local_tp_ctx;
                    join_params.moe_runtime_table = binding_it->second.moe_runtime_table;
                    join_params.tp_device_idx = config_.tp_device_idx;
                    join_params.config = binding_it->second.config;
                    join_params.local_transfer_slots =
                        transfer_directory_it->second->deviceEntries();
                    join_params.local_transfer_slot_count =
                        transfer_directory_it->second->slotCount();
                    join_params.collective_payload_slot_bytes =
                        binding_it->second.collective_payload_slot_bytes;
                    join_params.collective_payload_slot_capacity =
                        binding_it->second.collective_payload_slot_capacity;
                    join_params.stage_name = join_node;
                    join_params.workspace_name = binding_it->second.workspace_name;
                    join_params.phase = DeviceMoERebalanceStagePhase::JoinTransfer;
                    join_params.transfer_mode = binding_it->second.transfer_mode;
                    join_params.transfer_state = transfer_state_it->second;

                    graph.addNode(join_node,
                                  ComputeStageFactory::createMoEDeviceRebalance(join_params),
                                  device);
                    graph.addDependency(join_node, ffn_terminal);
                    ffn_terminal = join_node;
                }
            }
        }

        if (!mtp_sidecar_context &&
            device.is_gpu() &&
            mirroredLayerDiagnosticsEnabled() &&
            !ffn_terminal.empty())
        {
            /*
             * Preserve the two values needed to reconstruct a transformer
             * layer boundary in this graph family:
             *
             *  - current_hidden is the accumulated residual after attention.
             *  - attn_proj is the current layer's FFN delta.
             *
             * The next layer's fused residual norm combines those tensors.
             * Capturing both therefore distinguishes an attention/residual
             * divergence from a routed/shared-FFN divergence without copying a
             * complete activation matrix. HiddenStateRowSelectStage records
             * only the final logical row, so storage remains
             * O(graph variants * layers * d_model) even when a request
             * contains thousands of prefill or verifier rows.
             *
             * Grouped verifier graphs need their own bank. Accepted-state
             * publication can select the next terminal hidden row directly
             * from a grouped verifier result; retaining only the older prefill
             * bank would hide the producer that actually supplied the failed
             * mailbox payload.
             */
            const std::string graph_regime =
                config_.grouped_mtp_verifier
                    ? "grouped_verifier"
                    : config_.live_mtp_request_batch_condition
                    ? "request_batch_decode"
                    : total_tokens == 1
                          ? "serial_decode"
                          : "prefill";
            const std::string graph_shape =
                graph_regime + "_m" + std::to_string(total_tokens);

            auto add_row_checkpoint =
                [&](const std::string &boundary,
                    const ITensor *source,
                    BufferId source_buffer_id,
                    std::optional<int> fixed_row,
                    const std::string &dependency) -> std::string
            {
                if (!source)
                {
                    throw std::runtime_error(
                        "Qwen35 MoE mirrored-layer diagnostics lost source tensor for " +
                        boundary);
                }

                const std::string checkpoint_name =
                    graph_shape + "_layer" + std::to_string(layer_idx) +
                    "_" + boundary;
                auto &checkpoint = mirrored_layer_checkpoints_[checkpoint_name];
                if (!checkpoint)
                {
                    checkpoint = std::make_unique<FP32Tensor>(
                        std::vector<size_t>{
                            1u,
                            static_cast<size_t>(config_.d_model)},
                        device);
                    /*
                     * Diagnostic checkpoints are graph outputs, not host
                     * tensors awaiting upload. Prepare their model-lifetime
                     * device storage before the executor validates output
                     * placement. This allocation exists only when the
                     * opt-in mirrored-layer diagnostic changes graph topology;
                     * normal inference never creates these tensors.
                     */
                    if (!checkpoint->allocateOnDevice(device))
                    {
                        throw std::runtime_error(
                            "Qwen35 MoE could not allocate mirrored-layer "
                            "diagnostic checkpoint " +
                            checkpoint_name + " on " + device.to_string());
                    }
                }

                const std::string node_name =
                    prefix + "mirror_checkpoint_" + graph_shape + "_" + boundary;
                HiddenStateRowSelectStage::Params checkpoint_params;
                checkpoint_params.device_id = device;
                checkpoint_params.input = source;
                checkpoint_params.output = checkpoint.get();
                checkpoint_params.input_buffer_id = source_buffer_id;
                checkpoint_params.seq_len = total_tokens;
                checkpoint_params.d_model = config_.d_model;
                if (fixed_row)
                {
                    checkpoint_params.selected_row_idx = *fixed_row;
                    checkpoint_params.selection_policy =
                        HiddenStateRowSelectStage::SelectionPolicy::
                            FixedDeviceRow;
                }
                else
                {
                    checkpoint_params.selected_row_idx = total_tokens - 1;
                    configureMirroredCheckpointRowOwnership(
                        checkpoint_params,
                        total_tokens,
                        device,
                        sequence_lengths_device);
                }

                graph.addNode(
                    node_name,
                    ComputeStageFactory::createHiddenStateRowSelect(
                        checkpoint_params),
                    device);
                graph.addDependency(node_name, dependency);
                return node_name;
            };

            const auto add_layer_boundary_bank =
                [&](const std::string &boundary,
                    const ITensor *source,
                    BufferId source_buffer_id,
                    const std::string &dependency) -> std::string
            {
                std::string checkpoint_dependency = add_row_checkpoint(
                    boundary,
                    source,
                    source_buffer_id,
                    std::nullopt,
                    dependency);
                if (!config_.grouped_mtp_verifier)
                    return checkpoint_dependency;

                /*
                 * A later transaction can fail in row zero even when the
                 * grouped graph's terminal/bonus row remains correct. Keep
                 * every physical verifier row under this opt-in diagnostic so
                 * the first divergent transformer boundary is observable in
                 * one run instead of requiring one recapture per layer/row.
                 */
                for (int row = 0; row < total_tokens; ++row)
                {
                    checkpoint_dependency = add_row_checkpoint(
                        boundary + "_row" + std::to_string(row),
                        source,
                        source_buffer_id,
                        row,
                        checkpoint_dependency);
                }
                return checkpoint_dependency;
            };

            const std::string hidden_checkpoint =
                add_layer_boundary_bank(
                    "attention_residual",
                    buffers.current_hidden,
                    BufferId::HIDDEN_STATE,
                    ffn_terminal);
            ffn_terminal =
                add_layer_boundary_bank(
                    "ffn_delta",
                    buffers.attn_proj,
                    BufferId::ATTN_PROJ,
                    hidden_checkpoint);
        }

        graph.setTerminalNode(ffn_terminal);
        return graph;
    }

    std::vector<Qwen35MoEGraph::MirroredLayerCheckpoint>
    Qwen35MoEGraph::mirroredLayerCheckpoints() const
    {
        std::vector<MirroredLayerCheckpoint> checkpoints;
        checkpoints.reserve(mirrored_layer_checkpoints_.size());
        for (const auto &[name, tensor] : mirrored_layer_checkpoints_)
        {
            if (!tensor)
                continue;
            checkpoints.push_back({
                .name = name,
                .tensor = tensor.get(),
            });
        }
        std::sort(
            checkpoints.begin(),
            checkpoints.end(),
            [](const MirroredLayerCheckpoint &lhs,
               const MirroredLayerCheckpoint &rhs)
            {
                return lhs.name < rhs.name;
            });
        return checkpoints;
    }

    std::string Qwen35MoEGraph::maybeAddEmbeddingDiagnosticCheckpoints(
        ComputeGraph &graph,
        TensorBase *source,
        BufferId source_buffer_id,
        const std::string &dependency,
        int total_tokens,
        DeviceId device)
    {
        if (mtpGraphContextActive() ||
            !device.is_gpu() ||
            !config_.grouped_mtp_verifier ||
            !mirroredLayerDiagnosticsEnabled())
        {
            return dependency;
        }
        if (!source || dependency.empty() || total_tokens <= 0)
        {
            throw std::runtime_error(
                "Qwen35 MoE grouped-verifier embedding diagnostics require "
                "an exact source, producer, and positive row count");
        }

        const std::string graph_shape =
            "grouped_verifier_m" + std::to_string(total_tokens);
        std::string checkpoint_dependency = dependency;
        for (int row = 0; row < total_tokens; ++row)
        {
            const std::string checkpoint_name =
                graph_shape + "_embedding_row" + std::to_string(row);
            auto &checkpoint =
                mirrored_layer_checkpoints_[checkpoint_name];
            if (!checkpoint)
            {
                checkpoint = std::make_unique<FP32Tensor>(
                    std::vector<size_t>{
                        1u,
                        static_cast<size_t>(config_.d_model)},
                    device);
                if (!checkpoint->allocateOnDevice(device))
                {
                    throw std::runtime_error(
                        "Qwen35 MoE could not allocate grouped-verifier "
                        "embedding checkpoint " +
                        checkpoint_name + " on " + device.to_string());
                }
            }

            const std::string node_name =
                "mirror_checkpoint_" + checkpoint_name;
            HiddenStateRowSelectStage::Params params;
            params.device_id = device;
            params.input = source;
            params.output = checkpoint.get();
            params.input_buffer_id = source_buffer_id;
            params.seq_len = total_tokens;
            params.d_model = config_.d_model;
            params.selected_row_idx = row;
            params.selection_policy =
                HiddenStateRowSelectStage::SelectionPolicy::FixedDeviceRow;

            graph.addNode(
                node_name,
                ComputeStageFactory::createHiddenStateRowSelect(params),
                device);
            graph.addDependency(node_name, checkpoint_dependency);
            checkpoint_dependency = node_name;
        }
        return checkpoint_dependency;
    }

    std::string Qwen35MoEGraph::maybeAddGDNDiagnosticCheckpoint(
        ComputeGraph &graph,
        const std::string &boundary,
        const ITensor *source,
        BufferId source_buffer_id,
        const std::string &dependency,
        int layer_idx,
        int total_tokens,
        int feature_dim,
        DeviceId device,
        const int32_t *sequence_lengths_device)
    {
        if (mtpGraphContextActive() ||
            !device.is_gpu() ||
            !mirroredLayerDiagnosticsEnabled())
        {
            return dependency;
        }
        if (!source || layer_idx < 0 ||
            total_tokens <= 0 || feature_dim <= 0)
        {
            throw std::runtime_error(
                "Qwen35 MoE GDN diagnostics require a source tensor and "
                "positive layer, row, and feature geometry");
        }

        const std::string graph_regime =
            config_.grouped_mtp_verifier
                ? "grouped_verifier"
                : config_.live_mtp_request_batch_condition
                ? "request_batch_decode"
                : total_tokens == 1
                      ? "serial_decode"
                      : "prefill";
        const std::string graph_shape =
            graph_regime + "_m" + std::to_string(total_tokens);
        const auto add_checkpoint =
            [&](const std::string &checkpoint_suffix,
                int selected_row,
                HiddenStateRowSelectStage::SelectionPolicy selection_policy,
                const int32_t *request_lengths,
                const std::string &input_dependency) -> std::string
        {
            const std::string checkpoint_name =
                graph_shape + "_layer" + std::to_string(layer_idx) +
                "_" + boundary + checkpoint_suffix;
            auto &checkpoint =
                mirrored_layer_checkpoints_[checkpoint_name];
            if (!checkpoint)
            {
                checkpoint = std::make_unique<FP32Tensor>(
                    std::vector<size_t>{
                        1u,
                        static_cast<size_t>(feature_dim)},
                    device);
                if (!checkpoint->allocateOnDevice(device))
                {
                    throw std::runtime_error(
                        "Qwen35 MoE could not allocate GDN diagnostic checkpoint " +
                        checkpoint_name + " on " + device.to_string());
                }
            }

            const std::string node_name =
                "layer" + std::to_string(layer_idx) +
                "_mirror_checkpoint_" + graph_shape + "_" + boundary +
                checkpoint_suffix;
            HiddenStateRowSelectStage::Params params;
            params.device_id = device;
            params.input = source;
            params.output = checkpoint.get();
            params.input_buffer_id = source_buffer_id;
            params.seq_len = total_tokens;
            params.d_model = feature_dim;
            params.selected_row_idx = selected_row;
            params.selection_policy = selection_policy;
            params.request_sequence_length_device = request_lengths;

            graph.addNode(
                node_name,
                ComputeStageFactory::createHiddenStateRowSelect(params),
                device);
            graph.addDependency(node_name, input_dependency);
            return node_name;
        };

        HiddenStateRowSelectStage::Params terminal_policy;
        terminal_policy.selected_row_idx = total_tokens - 1;
        configureMirroredCheckpointRowOwnership(
            terminal_policy,
            total_tokens,
            device,
            sequence_lengths_device);
        std::string checkpoint_dependency = add_checkpoint(
            "",
            terminal_policy.selected_row_idx,
            terminal_policy.selection_policy,
            terminal_policy.request_sequence_length_device,
            dependency);

        /*
         * Fixed rows are diagnostic observations of immutable graph geometry.
         * They deliberately do not consume mutable request-length metadata.
         * Keep the terminal checkpoint above device-owned so padded prefill
         * still identifies the exact logical final row.
         */
        const auto selected_layer = mirroredGDNDiagnosticLayer();
        if (selected_layer && *selected_layer == layer_idx &&
            graph_regime == "grouped_verifier")
        {
            /*
             * A grouped verifier graph is captured at its physical M, but the
             * final response budget may make only a strict prefix logical work.
             * Retain every physical row for the selected layer so diagnostics
             * can compare all valid rows and can also expose writes into padded
             * rows. The unsuffixed checkpoint above remains the authoritative
             * logical terminal row selected by resident request length.
             */
            for (int row = 0; row < total_tokens; ++row)
            {
                checkpoint_dependency = add_checkpoint(
                    "_row" + std::to_string(row),
                    row,
                    HiddenStateRowSelectStage::SelectionPolicy::FixedDeviceRow,
                    nullptr,
                    checkpoint_dependency);
            }
        }
        else if (graph_regime == "prefill" &&
                 selected_layer &&
                 *selected_layer == layer_idx)
        {
            static constexpr int kSampleRows[] = {
                0, 1, 2, 7, 31, 127, 511,
                575, 639, 703, 767, 831, 895, 959, 1023,
                2047};
            for (const int row : kSampleRows)
            {
                if (row >= total_tokens)
                    continue;
                checkpoint_dependency = add_checkpoint(
                    "_row" + std::to_string(row),
                    row,
                    HiddenStateRowSelectStage::SelectionPolicy::FixedDeviceRow,
                    nullptr,
                    checkpoint_dependency);
            }
        }
        return checkpoint_dependency;
    }

    std::string Qwen35MoEGraph::maybeAddFinalNormDiagnosticCheckpoint(
        ComputeGraph &graph,
        const std::string &boundary,
        TensorBase *source,
        BufferId source_buffer_id,
        const std::string &dependency,
        int total_tokens,
        DeviceId device,
        const int32_t *sequence_lengths_device)
    {
        if (mtpGraphContextActive() ||
            !device.is_gpu() ||
            !mirroredLayerDiagnosticsEnabled())
        {
            return dependency;
        }
        if (!source || total_tokens <= 0)
        {
            throw std::runtime_error(
                "Qwen35 MoE final-norm diagnostics require a source tensor "
                "and at least one logical row");
        }

        const std::string graph_regime =
            config_.grouped_mtp_verifier
                ? "grouped_verifier"
                : config_.live_mtp_request_batch_condition
                ? "request_batch_decode"
                : total_tokens == 1
                      ? "serial_decode"
                      : "prefill";
        const std::string graph_shape =
            graph_regime + "_m" + std::to_string(total_tokens);
        const std::string checkpoint_name =
            graph_shape + "_" + boundary;
        auto &checkpoint =
            mirrored_layer_checkpoints_[checkpoint_name];
        if (!checkpoint)
        {
            checkpoint = std::make_unique<FP32Tensor>(
                std::vector<size_t>{
                    1u,
                    static_cast<size_t>(config_.d_model)},
                device);
            /*
             * See the per-layer checkpoint allocation above. Final-norm
             * diagnostics are graph outputs as well and therefore require
             * prepared device storage before execution begins.
             */
            if (!checkpoint->allocateOnDevice(device))
            {
                throw std::runtime_error(
                    "Qwen35 MoE could not allocate final-norm diagnostic "
                    "checkpoint " +
                    checkpoint_name + " on " + device.to_string());
            }
        }

        const std::string node_name =
            "mirror_checkpoint_" + checkpoint_name;
        HiddenStateRowSelectStage::Params params;
        params.device_id = device;
        params.input = source;
        params.output = checkpoint.get();
        params.input_buffer_id = source_buffer_id;
        params.seq_len = total_tokens;
        params.d_model = config_.d_model;
        params.selected_row_idx = total_tokens - 1;
        configureMirroredCheckpointRowOwnership(
            params,
            total_tokens,
            device,
            sequence_lengths_device);

        graph.addNode(
            node_name,
            ComputeStageFactory::createHiddenStateRowSelect(params),
            device);
        if (!dependency.empty())
            graph.addDependency(node_name, dependency);
        return node_name;
    }

    HiddenStateRowSelectStage::SelectionPolicy
    Qwen35MoEGraph::mirroredCheckpointSelectionPolicy(
        int total_tokens,
        DeviceId device,
        const int32_t *sequence_lengths_device) const
    {
        /*
         * Both ordinary prefill and grouped verification may execute a strict
         * logical prefix of their captured physical M. Prefill does so through
         * bucket reuse; a grouped verifier does so when the remaining response
         * budget is smaller than the configured proposal depth. In either case,
         * the persistent device request length is the sole terminal-row owner.
         * Refusing to construct a multi-row GPU checkpoint without that owner
         * makes padded-row selection structurally impossible.
         */
        const bool resident_length_owned =
            device.is_gpu() &&
            total_tokens > 1 &&
            (config_.grouped_mtp_verifier ||
             !config_.live_mtp_request_batch_condition);
        if (!resident_length_owned)
        {
            return HiddenStateRowSelectStage::SelectionPolicy::
                FixedDeviceRow;
        }
        if (!sequence_lengths_device)
        {
            throw std::runtime_error(
                "Qwen35 MoE mirrored multi-row checkpoint requires a "
                "device-resident request length");
        }
        return HiddenStateRowSelectStage::SelectionPolicy::
            DeviceResidentRequestLength;
    }

    void Qwen35MoEGraph::configureMirroredCheckpointRowOwnership(
        HiddenStateRowSelectStage::Params &params,
        int total_tokens,
        DeviceId device,
        const int32_t *sequence_lengths_device) const
    {
        params.selection_policy =
            mirroredCheckpointSelectionPolicy(
                total_tokens,
                device,
                sequence_lengths_device);
        params.request_sequence_length_device =
            params.selection_policy ==
                    HiddenStateRowSelectStage::SelectionPolicy::
                        DeviceResidentRequestLength
                ? sequence_lengths_device
                : nullptr;
    }

} // namespace llaminar2
