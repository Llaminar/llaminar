/**
 * @file Qwen35MoEGraph.cpp
 * @brief Qwen 3.5 MoE compute graph builder implementation
 */

#include "Qwen35MoEGraph.h"
#include "Qwen35MoESchema.h"
#include "../../collective/ILocalTPContext.h"
#include "../../utils/Logger.h"
#include "../../execution/compute_stages/ComputeStageFactory.h"
#include "../../execution/compute_stages/stages/MoEExpertDispatchStage.h"
#include "../../execution/compute_stages/stages/MoERoutingStage.h"
#include "../../execution/compute_stages/stages/MoEExpertComputeStage.h"
#include "../../execution/compute_stages/stages/HiddenStateRowSelectStage.h"
#include "../../execution/compute_stages/stages/MoELocalExpertStage.h"
#include "../../execution/compute_stages/stages/MoESparseDispatchStage.h"
#include "../../execution/compute_stages/stages/MoESparseReturnReduceStage.h"
#include "../../execution/moe/MoERoutedExpertPlacementPlan.h"
#include "../../execution/moe/MoEExpertOwnerMap.h"
#include "../../execution/moe/MoEExpertOverlayRuntimePlan.h"
#include "../../execution/moe/MoEOverlaySparseCollective.h"
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
#include <memory>
#include <mutex>
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
        constexpr uint32_t kMoEPrefixRuntimeVersion = 3;

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

        std::vector<int> contiguousApportionedExpertOwners(
            int num_experts,
            int participant_count)
        {
            std::vector<int> owners(static_cast<size_t>(std::max(0, num_experts)), -1);
            if (num_experts <= 0 || participant_count <= 0)
                return owners;

            const int base = num_experts / participant_count;
            const int remainder = num_experts % participant_count;
            for (int participant = 0; participant < participant_count; ++participant)
            {
                const int count = base + (participant < remainder ? 1 : 0);
                const int start = participant * base + std::min(participant, remainder);
                for (int expert = start; expert < start + count && expert < num_experts; ++expert)
                    owners[static_cast<size_t>(expert)] = participant;
            }
            return owners;
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
                    !descriptor.gate.valid() ||
                    !descriptor.up.valid() ||
                    !descriptor.down.valid() ||
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
            const std::vector<int> &owner_participants = {})
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
                const bool expected_local = expertMaskEnablesExpert(expert_mask, expert, num_experts);
                if (bank.local_compute_mask[static_cast<size_t>(expert)] != (expected_local ? 1u : 0u))
                    return false;
                const int expected_owner =
                    owner_participants.empty()
                        ? (expected_local ? local_participant : -1)
                        : owner_participants[static_cast<size_t>(expert)];
                if (!owner_participants.empty() && expected_owner < 0)
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
                if (!expected_local)
                    continue;

                const auto &desc = bank.experts[static_cast<size_t>(expert)];
                if (desc.logical_expert_id != expert ||
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

        bool runtimeTableHasUsableLiveDynamicDecodeBank(
            IMoERuntimeTable *runtime_table,
            int layer_idx,
            int num_experts,
            int top_k,
            int local_participant,
            int participant_count,
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
                const auto &desc = bank.experts[static_cast<size_t>(expert)];
                const uint32_t raw_resident_mask =
                    bank.resident_participant_mask[static_cast<size_t>(expert)];
                uint32_t effective_resident_mask = raw_resident_mask & valid_participant_mask;
                const bool local_compute =
                    bank.local_compute_mask[static_cast<size_t>(expert)] != 0u;

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
                if (effective_resident_mask == 0u)
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
                    !desc.gate.valid() ||
                    !desc.up.valid() ||
                    !desc.down.valid() ||
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
            const std::vector<ITensorGemm *> &gate_gemms,
            const std::vector<ITensorGemm *> &up_gemms,
            const std::vector<ITensorGemm *> &down_gemms,
            void *stream,
            bool allow_existing_dynamic_bank,
            const std::string &context)
        {
            if (!runtime_table || layer_idx < 0)
                return false;
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

            if (runtimeTableHasUsableMaskedDecodeBank(
                    runtime_table,
                    layer_idx,
                    num_experts,
                    top_k,
                    expert_mask,
                    local_participant,
                    participant_count,
                    owner_participants))
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

            bool has_local_expert = false;
            for (int expert = 0; expert < num_experts; ++expert)
            {
                const int expert_owner =
                    owner_participants.empty()
                        ? (expertMaskEnablesExpert(expert_mask, expert, num_experts) ? local_participant : -1)
                        : owner_participants[static_cast<size_t>(expert)];
                if (!owner_participants.empty() && expert_owner < 0)
                {
                    LOG_ERROR("[Qwen35MoEGraph] " << context
                                                  << ": missing owner participant for expert "
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
                    update.experts[static_cast<size_t>(expert)].logical_expert_id = expert;
                    update.experts[static_cast<size_t>(expert)].owner_participant = expert_owner;
                    continue;
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
                if (!gate->exportNativeVNNIMatrixDesc(desc.gate) ||
                    !up->exportNativeVNNIMatrixDesc(desc.up) ||
                    !down->exportNativeVNNIMatrixDesc(desc.down))
                {
                    LOG_ERROR("[Qwen35MoEGraph] " << context
                                                  << ": prepared GEMM engines for local expert "
                                                  << expert << " layer " << layer_idx
                                                  << " cannot export native-VNNI descriptors");
                    return false;
                }

                if (desc.gate.n != expert_intermediate || desc.gate.k != d_model ||
                    desc.up.n != expert_intermediate || desc.up.k != d_model ||
                    desc.down.n != d_model || desc.down.k != expert_intermediate)
                {
                    LOG_ERROR("[Qwen35MoEGraph] " << context
                                                  << ": native-VNNI descriptor shape mismatch for local expert "
                                                  << expert << " layer " << layer_idx);
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
                owner_participants);
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
            fields.push_back({prefix + ".routed_assignment_policy", routedExpertAssignmentPolicyToString(domain.routed_assignment_policy)});
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
            fields.push_back({prefix + ".routed_assignment_policy", routedExpertAssignmentPolicyToString(domain.routed_assignment_policy)});
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
                fields.push_back({prefix + ".scope", executionDomainScopeToString(domain.scope)});
                fields.push_back({prefix + ".backend", collectiveBackendTypeToString(domain.backend)});
                fields.push_back({prefix + ".routed_compute_policy", routedExpertComputePolicyToString(domain.routed_compute_policy)});
                fields.push_back({prefix + ".routed_phase_policy", routedExpertPhasePolicyToString(domain.routed_phase_policy)});
                fields.push_back({prefix + ".routed_assignment_policy", routedExpertAssignmentPolicyToString(domain.routed_assignment_policy)});
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
                   domain->scope == ExecutionDomainScope::LOCAL &&
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
                   domain->scope == ExecutionDomainScope::LOCAL &&
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
            if (!plan.isTieredOverlay() ||
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
            if (!plan.isTieredOverlay() ||
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

        int parsedLLEPPrefillTransferMode()
        {
            const int mode = debugEnv().moe_rebalance.llep_prefill_transfer_mode;
            if (mode < 0 || mode > 1)
            {
                throw std::runtime_error(
                    "Invalid LLAMINAR_MOE_LLEP_PREFILL_TRANSFER_MODE "
                    "(valid: resident-only, full)");
            }
            return mode;
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
                if (!engine->exportNativeVNNIMatrixDesc(descriptor) ||
                    !descriptor.valid() ||
                    descriptor.n != n ||
                    descriptor.k != k)
                {
                    return std::nullopt;
                }

                uint8_t payload_bytes_per_block = 0;
                uint8_t is_asymmetric = 0;
                uint8_t has_emins = 0;
                if (!deviceMoEProjectionFormat(
                        descriptor,
                        payload_bytes_per_block,
                        is_asymmetric,
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
                spec.is_asymmetric = is_asymmetric != 0;
                spec.has_emins = has_emins != 0;
                spec.codebook_id = descriptor.codebook_id;
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

    void Qwen35MoEGraph::resetState(void *execution_stream)
    {
        Qwen35Graph::resetState(execution_stream);
        prefix_runtime_device_rehydration_pending_ = false;

        if (config_.moe.decode_histogram)
            config_.moe.decode_histogram->resetWindow();

        for (auto &[key, table] : moe_runtime_tables_)
        {
            (void)key;
            if (table)
                table->restoreInitialRuntimeState(execution_stream);
        }

        /*
         * Runtime-table placement and transfer-directory occupancy are one
         * publication transaction. The baseline table never references a
         * rolling transfer slot, so retire every directory occupant on the
         * same reset stream before RequestStateResetReady can be recorded.
         */
        for (auto &[key, directory] : moe_transfer_slot_directories_)
        {
            (void)key;
            if (directory)
                directory->resetRequestPublications(execution_stream);
        }
    }

    void Qwen35MoEGraph::resetPrefixCacheRuntimeStateWithoutSnapshot(
        void *execution_stream)
    {
        /*
         * A cache block without a portable MoE payload owns no request-local
         * placement changes. Restore the immutable model-lifetime placement
         * template captured during cold graph construction. The template uses
         * the same runtime-table and scratch addresses as every captured graph,
         * so content restoration cannot require recapture.
         */
        Qwen35Graph::resetState(execution_stream);
        prefix_runtime_device_rehydration_pending_ = false;

        if (config_.moe.decode_histogram)
            config_.moe.decode_histogram->resetWindow();

        /*
         * Graph-side bindings, auxiliary streams, and transfer-slot
         * directories own model-lifetime pointer identities captured by stage
         * nodes. Prefix reset must preserve those owners. Runtime placement
         * claims are request state; restoreInitialRuntimeState() removes every
         * transient arrival claim while retaining the stable directory
         * addresses and canonical static descriptors that the existing graph
         * replay will use.
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
            if (table)
                table->restoreInitialRuntimeState(execution_stream);
        }
        for (auto &[key, directory] : moe_transfer_slot_directories_)
        {
            (void)key;
            if (directory)
                directory->resetRequestPublications(execution_stream);
        }
    }

    ILocalTPContext *Qwen35MoEGraph::maintenanceTPContextForDomain(
        const std::string &domain_key,
        ILocalTPContext &decode_tp_ctx)
    {
        auto &maintenance_ctx = moe_maintenance_tp_contexts_[domain_key];
        if (maintenance_ctx)
            return maintenance_ctx.get();

        static std::mutex registry_mutex;
        static std::unordered_map<std::string, std::weak_ptr<ILocalTPContext>>
            registry;

        std::lock_guard<std::mutex> lock(registry_mutex);
        if (auto existing = registry[domain_key].lock())
        {
            if (existing->degree() != decode_tp_ctx.degree() ||
                existing->backend() != decode_tp_ctx.backend())
            {
                throw std::runtime_error(
                    "Qwen35 MoE graph-side rebalance reused an incompatible maintenance lane for " +
                    domain_key);
            }
            maintenance_ctx = std::move(existing);
            LOG_INFO("[Qwen35MoEGraph] Reusing dedicated MoE rebalance maintenance collective lane"
                     << " domain=" << domain_key
                     << " degree=" << maintenance_ctx->degree()
                     << " backend=" << collectiveBackendTypeToString(maintenance_ctx->backend())
                     << " decode_ctx=" << static_cast<const void *>(&decode_tp_ctx)
                     << " maintenance_ctx=" << static_cast<const void *>(maintenance_ctx.get()));
            return maintenance_ctx.get();
        }

        auto created =
            createLocalTPContext(
                decode_tp_ctx.devices(),
                decode_tp_ctx.weights(),
                decode_tp_ctx.backend());
        if (!created)
        {
            throw std::runtime_error(
                "Qwen35 MoE graph-side rebalance could not create maintenance collective lane for " +
                domain_key);
        }
        if (created->degree() != decode_tp_ctx.degree() ||
            created->backend() != decode_tp_ctx.backend())
        {
            throw std::runtime_error(
                "Qwen35 MoE graph-side rebalance maintenance lane does not match decode lane for " +
                domain_key);
        }

        maintenance_ctx = std::shared_ptr<ILocalTPContext>(std::move(created));
        registry[domain_key] = maintenance_ctx;
        LOG_INFO("[Qwen35MoEGraph] Created dedicated MoE rebalance maintenance collective lane"
                 << " domain=" << domain_key
                 << " degree=" << maintenance_ctx->degree()
                 << " backend=" << collectiveBackendTypeToString(maintenance_ctx->backend())
                 << " decode_ctx=" << static_cast<const void *>(&decode_tp_ctx)
                 << " maintenance_ctx=" << static_cast<const void *>(maintenance_ctx.get()));
        return maintenance_ctx.get();
    }

    /**
     * @brief Locate the domain-wide decode maintenance binding for a device.
     *
     * The graph builder creates two different classes of graph-side rebalance
     * binding when long-context prefix-cache MTP runs with phase-split LLEP:
     * a decode-maintenance binding and one or more layer-local prefill LLEP
     * transfer bindings.  The async maintenance graph is a decode-time control
     * loop, so it must use the decode binding even when a prefill binding was
     * inserted later into the unordered binding map.
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
        if (!binding.decode_tp_ctx ||
            !binding.maintenance_tp_ctx ||
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
        params.tp_ctx = binding.maintenance_tp_ctx;
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

    bool Qwen35MoEGraph::restorePrefixCacheRuntimeState(const std::vector<uint8_t> &state, void *stream)
    {
        prefix_runtime_device_rehydration_pending_ = false;
        if (state.empty())
            return true;
        if (state.size() < sizeof(kMoEPrefixRuntimeMagic) ||
            std::memcmp(state.data(), kMoEPrefixRuntimeMagic, sizeof(kMoEPrefixRuntimeMagic)) != 0)
        {
            LOG_ERROR("[Qwen35MoEGraph] Refusing obsolete prefix-cache MoE runtime state: "
                      "MoE placement snapshots must not serialize runtime-local device descriptors");
            return false;
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
            return false;
        }
        if (version != kMoEPrefixRuntimeVersion)
        {
            LOG_ERROR("[Qwen35MoEGraph] Unsupported prefix-cache MoE runtime state version "
                      << version);
            return false;
        }
        if (top_k != static_cast<uint32_t>(config_.moe.top_k))
        {
            LOG_ERROR("[Qwen35MoEGraph] Prefix-cache MoE histogram top-k mismatch: blob="
                      << top_k << " graph=" << config_.moe.top_k);
            return false;
        }

        uint32_t restored_tables = 0;
        bool requires_device_rehydration = false;
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
                return false;
            }
            if (experts != static_cast<uint32_t>(config_.moe.num_experts))
            {
                LOG_ERROR("[Qwen35MoEGraph] Prefix-cache MoE portable runtime expert-count mismatch for "
                          << key << ": blob=" << experts
                          << " graph=" << config_.moe.num_experts);
                return false;
            }
            auto it = moe_runtime_tables_.find(key);
            if (it == moe_runtime_tables_.end() || !it->second)
            {
                LOG_ERROR("[Qwen35MoEGraph] Prefix-cache MoE portable runtime restore could not find runtime table "
                          << key);
                return false;
            }
            if (layers != static_cast<uint32_t>(it->second->layerCount()))
            {
                LOG_ERROR("[Qwen35MoEGraph] Prefix-cache MoE portable runtime layer-count mismatch for "
                          << key << ": blob=" << layers
                          << " table=" << it->second->layerCount());
                return false;
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
                    return false;
                }
                if (layer.expert_count != experts ||
                    layer.top_k != static_cast<uint32_t>(config_.moe.top_k))
                {
                    LOG_ERROR("[Qwen35MoEGraph] Prefix-cache MoE portable runtime layer metadata mismatch for "
                              << key << " layer=" << layer_idx);
                    return false;
                }
                requires_device_rehydration =
                    requires_device_rehydration ||
                    layer.requires_device_payload_rehydration != 0u;
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
                        return false;
                    }
                    expert.logical_expert_id = logical_expert_id;
                    expert.owner_participant = owner_participant;
                    expert.local_slot = local_slot;
                    expert.local_compute = local_compute != 0u ? 1u : 0u;
                    expert.replica_role = static_cast<uint8_t>(replica_role);
                }
                layer.selected_histogram.assign(static_cast<size_t>(experts), 0ULL);
                layer.local_histogram.assign(static_cast<size_t>(experts), 0ULL);
                for (uint32_t expert_idx = 0; expert_idx < experts; ++expert_idx)
                {
                    if (!readU64(state, offset, layer.selected_histogram[static_cast<size_t>(expert_idx)]))
                    {
                        LOG_ERROR("[Qwen35MoEGraph] Malformed prefix-cache MoE portable selected histogram payload");
                        return false;
                    }
                }
                for (uint32_t expert_idx = 0; expert_idx < experts; ++expert_idx)
                {
                    if (!readU64(state, offset, layer.local_histogram[static_cast<size_t>(expert_idx)]))
                    {
                        LOG_ERROR("[Qwen35MoEGraph] Malformed prefix-cache MoE portable local histogram payload");
                        return false;
                    }
                }
            }

            /*
             * Portable version 3 contains pointer-free logical placement.
             * Transfer-slot descriptors are reconstructed from immutable owner
             * payloads by the dedicated captured rehydration transaction; blob
             * parsing must never consult a rolling slot directory whose bytes
             * may have been reused since the prefix was harvested.
             */
            if (!it->second->restorePortableRuntimeState(runtime_layers, stream))
            {
                LOG_ERROR("[Qwen35MoEGraph] Prefix-cache MoE portable runtime restore failed for " << key);
                return false;
            }
            ++restored_tables;
        }
        if (offset != state.size())
        {
            LOG_ERROR("[Qwen35MoEGraph] Prefix-cache MoE runtime state has trailing bytes");
            return false;
        }

        prefix_runtime_device_rehydration_pending_ =
            requires_device_rehydration;
        LOG_INFO("[Qwen35MoEGraph] Prefix-runtime restore transaction"
                 << " device=" << config_.default_device.toString()
                 << " tables=" << restored_tables
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
             {"bytes", std::to_string(state.size())}});
        return true;
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

        /*
         * Graph executables freeze direct route-scratch pointers. Allocate the
         * complete serial execution domain before the first graph is built:
         * ordinary prefill needs the largest configured graph bucket, while
         * grouped verification and request-batched MTP need their flattened
         * target-row capacity. Every role is event/stream ordered by the
         * orchestrator, so one largest-participant arena is both correct and
         * substantially smaller than one allocation per layer and MTP depth.
         */
        const int prefill_graph_rows =
            resolveActivationBufferSeqLen(config_.max_seq_len, device);
        const int verifier_rows =
            config_.mtp.enabled
                ? resolveMTPMaxTargetQueryRows(config_.mtp)
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

        const std::string key = key_suffix.empty()
                                    ? device.to_string()
                                    : device.to_string() + "#" + key_suffix;
        auto it = moe_runtime_tables_.find(key);
        if (it != moe_runtime_tables_.end())
        {
            if (prefill_token_capacity > 0)
                it->second->ensurePrefillRouteScratchCapacity(prefill_token_capacity);
            registerRuntimeTableHistogramSyncIfNeeded(
                key,
                it->second.get(),
                register_decode_histogram);
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
        table_config.prefill_token_capacity = planned_route_rows;
        table_config.deferred_verifier_token_capacity =
            key_suffix.empty() && config_.mtp.enabled ? verifier_rows : 0;
        table_config.serial_route_scratch_arena = scratch_it->second;

        auto table = std::make_unique<MoERuntimeTable>(table_config);
        IMoERuntimeTable *ptr = table.get();
        registerRuntimeTableHistogramSyncIfNeeded(
            key,
            ptr,
            register_decode_histogram);
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

    void Qwen35MoEGraph::registerRuntimeTableHistogramSyncIfNeeded(
        const std::string &key,
        IMoERuntimeTable *table,
        bool register_decode_histogram)
    {
        if (!register_decode_histogram || !table || !config_.moe.decode_histogram)
            return;

        auto *histogram = config_.moe.decode_histogram;
        const std::string sync_key =
            key + "@" + std::to_string(reinterpret_cast<std::uintptr_t>(histogram));
        if (!moe_runtime_histogram_sync_keys_.insert(sync_key).second)
            return;

        histogram->registerRuntimeHistogramSync(
            [table, histogram]()
            {
                void *stream = table->decodeHistogramProducerStream();
                if (!stream)
                    throw std::runtime_error(
                        "[Qwen35MoEGraph] runtime histogram sync requested before a decode producer stream was recorded");
                return table->syncDecodeHistogramToHost(*histogram, stream);
            });
        PerfStatsCollector::addCounter(
            "moe_rebalance",
            "runtime_histogram_sync_registrations",
            1.0,
            "graph",
            "",
            {{"key", key}});
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
            config_.mtp.enabled ? std::max(1, resolveMTPMaxTargetQueryRows(config_.mtp)) : 1;
        const size_t moe_activation_rows = static_cast<size_t>(
            std::max(std::max(1, seq_len), mtp_target_query_rows));

        config.custom_formulas["moe_top_k"] = static_cast<size_t>(top_k);
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

        // If this layer doesn't have MoE weights, fall back to dense FFN
        if (!layer.moe_gate || !layer.moe_gate_exps)
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

        ComputeGraph graph;
        const bool mtp_sidecar_context = mtp_graph_context_active_;
        const int mtp_depth_idx = mtp_graph_depth_idx_;
        std::string prefix = mtp_sidecar_context
                                 ? "MTP" + std::to_string(std::max(0, mtp_depth_idx)) + "_"
                                 : "layer" + std::to_string(layer_idx) + "_";
        std::string ffn_terminal;
        int total_tokens = batch_size * seq_len;
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

        const bool overlay_requested = overlay_plan && overlay_plan->isTieredOverlay();
        const RoutedExpertLayerPlacement *overlay_placement = overlay_requested
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
                   config_.compute_all_position_logits &&
                   !mtp_sidecar_context;
        };
        auto forceDecodeEquivalentMoEVerifier = [&](DeviceId candidate)
        {
            return candidate.is_cpu() &&
                   total_tokens >= 1 &&
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
        const std::string runtime_table_suffix = use_mtp_runtime_table
                                                     ? "mtp_depth" + std::to_string(mtp_depth_idx)
                                                     : std::string{};
        const bool register_runtime_histogram = !use_mtp_runtime_table;
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
            config_.compute_all_position_logits &&
            layer_idx >= config_.pp_layer_offset &&
            layer_idx < config_.pp_layer_offset + config_.n_layers;
        const bool device_rebalance_decode_layer =
            local_decode_layer || grouped_main_verifier_layer;
        const bool first_device_rebalance_decode_layer =
            device_rebalance_decode_layer &&
            layer_idx == config_.pp_layer_offset;
        const bool last_device_rebalance_decode_layer =
            device_rebalance_decode_layer &&
            layer_idx == config_.pp_layer_offset + config_.n_layers - 1;
        auto *local_tp_ctx = dynamic_cast<ILocalTPContext *>(config_.tp_ctx);
        int hot_replica_cap = config_.moe.hot_expert_cache.resolveCap(
            config_.moe.num_experts,
            /*dynamic_rebalance_enabled=*/true);
        const auto &env = debugEnv();
        const int llep_prefill_transfer_mode =
            parsedLLEPPrefillTransferMode();
        const bool require_full_llep_prefill_transfer =
            llep_prefill_transfer_mode == 1;
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
            !mtp_sidecar_context &&
            total_tokens > 1 &&
            config_.moe.routed_assignment_policy == RoutedExpertAssignmentPolicy::LeastLoadedResident;
        const uint64_t llep_prefill_routed_rows =
            llep_prefill_requested
                ? static_cast<uint64_t>(std::max(0, total_tokens)) *
                      static_cast<uint64_t>(std::max(0, config_.moe.top_k))
                : 0ULL;
        const uint64_t llep_prefill_min_routed_rows =
            env.moe_rebalance.llep_prefill_min_routed_rows;
        const bool llep_prefill_cost_gate_passed =
            !llep_prefill_requested ||
            grouped_main_verifier_layer ||
            llep_prefill_min_routed_rows == 0ULL ||
            llep_prefill_routed_rows >= llep_prefill_min_routed_rows;
        const bool llep_prefill_enabled =
            llep_prefill_requested &&
            llep_prefill_cost_gate_passed &&
            local_tp_ctx != nullptr &&
            device.is_gpu();
        const bool llep_prefill_transport_supported =
            llep_prefill_enabled &&
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
        if (llep_prefill_enabled &&
            require_full_llep_prefill_transfer &&
            !grouped_main_verifier_layer &&
            !llep_prefill_transport_supported)
        {
            throw std::runtime_error(
                "Qwen35 MoE LLEP prefill transfer mode 'full' requires "
                "a homogeneous graph-capturable NCCL/RCCL LocalTP domain for " +
                device.to_string() + "; refusing resident-only or host fallback");
        }
        if (prefix_runtime_device_rehydration &&
            !prefix_runtime_rehydration_transport_supported)
        {
            throw std::runtime_error(
                "Qwen35 MoE prefix-runtime payload rehydration requires a homogeneous graph-capturable NCCL/RCCL LocalTP domain for " +
                device.to_string());
        }
        const RoutedExpertAssignmentPolicy prefill_routed_expert_assignment_policy =
            (total_tokens > 1 && !llep_prefill_enabled)
                ? RoutedExpertAssignmentPolicy::StaticOwner
                : config_.moe.routed_assignment_policy;
        if (env.presence.has("LLAMINAR_MOE_REBALANCE_REPLICAS"))
            hot_replica_cap = std::max(0, env.moe_rebalance.max_replicas);
        const bool device_side_graph_rebalance_candidate =
            device_rebalance_decode_layer &&
            config_.moe.rebalance_mode == MoERebalanceMode::DYNAMIC &&
            local_tp_ctx &&
            isHomogeneousGpuLocalTPRebalanceDomain(
                *local_tp_ctx,
                device,
                config_.tp_device_idx);
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
            (require_full_llep_prefill_transfer &&
             llep_prefill_transport_supported &&
             !grouped_main_verifier_layer);
        const bool prefill_llep_transfer_candidate =
            current_batch_llep_transfer_candidate ||
            prefix_runtime_rehydration_transport_supported;
        const bool graph_rebalance_transport_candidate =
            device_side_graph_rebalance_candidate ||
            prefill_llep_transfer_candidate;
        const bool register_runtime_histogram_for_decode =
            register_runtime_histogram &&
            !device_side_graph_rebalance_candidate;
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
                const int local_count = config_.moe.local_expert_count < 0
                                            ? config_.moe.num_experts
                                            : config_.moe.local_expert_count;
                if (config_.moe.local_expert_start != 0 ||
                    local_count != config_.moe.num_experts)
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
        const bool full_local_tp_replicated_overlay_decode_runtime_table =
            device.is_gpu() &&
            (total_tokens == 1 || forceGroupedMoEVerifierPrefill(device)) &&
            use_expert_overlay &&
            overlay_plan &&
            canUseLocalTPReplicatedFastPath(*overlay_plan, device);
        const bool masked_local_tp_apportioned_decode_runtime_table =
            (local_decode_layer ||
             (mtp_sidecar_context && total_tokens == 1)) &&
            device.is_gpu() &&
            !use_expert_overlay &&
            config_.moe.routed_compute_policy == RoutedExpertComputePolicy::Apportioned &&
            config_.moe.local_expert_count >= 0 &&
            local_tp_ctx &&
            local_tp_ctx->degree() > 1;
        const bool decode_runtime_table_eligible =
            static_full_local_expert_ownership ||
            masked_local_tp_overlay_decode_runtime_table ||
            full_local_tp_replicated_overlay_decode_runtime_table ||
            masked_local_tp_apportioned_decode_runtime_table;
        if (total_tokens == 1 &&
            rocm_env.moe_grouped_decode &&
            rocm_env.moe_device_routed_decode &&
            decode_runtime_table_eligible)
        {
            moe_runtime_table = moeRuntimeTableForDevice(
                device,
                total_tokens,
                runtime_table_suffix,
                runtime_table_layers,
                register_runtime_histogram_for_decode);
        }
        else if (total_tokens > 1)
        {
            // Fixed-topology grouped prefill consumes routing tensors directly.
            // Decode-equivalent verifier routing instead uses the same
            // runtime-table router as serial decode, so build the table even
            // if an old environment still tries to disable grouped prefill;
            // otherwise the strict verifier row proof would fail closed before
            // reaching the rows under test.
            // This table is prefill-only. It must not register as a decode
            // histogram source, because hot-cache overlay decode may use the
            // legacy host histogram path and never publish a runtime-table
            // decode producer stream.
            moe_runtime_table = moeRuntimeTableForDevice(
                device,
                total_tokens,
                runtime_table_suffix,
                runtime_table_layers,
                /*register_decode_histogram=*/false);
        }

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
            layer_formats.reserve(static_cast<size_t>(runtime_table_layers));

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
                 scan_layer < runtime_table_layers;
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
                << ":layers=" << runtime_table_layers
                << ":experts=" << config_.moe.num_experts
                << ":topk=" << config_.moe.top_k;
            return key.str();
        };
        auto graphRebalanceBindingKey = [&]() -> std::string
        {
            std::string key = graphRebalanceDomainKey();
            if (prefill_llep_transfer_candidate)
            {
                key += ":prefill_layer=";
                key += std::to_string(layer_idx);
            }
            return key;
        };
        auto graphRebalanceCollectiveKey = [&]() -> std::string
        {
            std::ostringstream key;
            key << "backend=" << static_cast<int>(local_tp_ctx ? local_tp_ctx->backend() : CollectiveBackendType::AUTO)
                << ":degree=" << (local_tp_ctx ? local_tp_ctx->degree() : 0)
                << ":layers=" << runtime_table_layers
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
        auto makeGraphRebalanceConfig = [&]() -> DeviceMoERebalanceConfig
        {
            auto deviceRebalanceConfigOrEnv =
                [&](uint32_t config_value, const char *env_name, int env_value) -> uint32_t
            {
                if (env.presence.has(env_name))
                    return static_cast<uint32_t>(std::max(0, env_value));
                return config_value;
            };

            DeviceMoERebalanceConfig rebalance_config;
            rebalance_config.num_layers = static_cast<uint32_t>(runtime_table_layers);
            rebalance_config.num_experts = static_cast<uint32_t>(config_.moe.num_experts);
            rebalance_config.top_k = static_cast<uint32_t>(config_.moe.top_k);
            rebalance_config.participant_id = static_cast<uint32_t>(config_.tp_device_idx);
            rebalance_config.participant_count = static_cast<uint32_t>(local_tp_ctx ? local_tp_ctx->degree() : 0);
            rebalance_config.root_participant = static_cast<uint32_t>(
                overlay_plan ? continuationRootParticipant(*overlay_plan) : 0);
            rebalance_config.window_size_tokens = static_cast<uint32_t>(
                std::max(1, config_.moe.rebalance_config.window_size));
            const int maintenance_slack =
                config_.moe.rebalance_config.device_maintenance_slack_tokens >= 0
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
                config_.moe.rebalance_config
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
                config_.moe.rebalance_config
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
                std::min(hot_replica_cap, config_.moe.num_experts));
            rebalance_config.dynamic_imbalance_threshold_per_mille =
                config_.moe.rebalance_config.dynamic_imbalance_threshold_per_mille;
            rebalance_config.dynamic_min_improvement_per_mille =
                config_.moe.rebalance_config.dynamic_min_improvement_per_mille;
            rebalance_config.dynamic_max_swaps_per_layer =
                config_.moe.rebalance_config.dynamic_max_swaps_per_layer;
            rebalance_config.dynamic_max_plan_entries_per_wave =
                config_.moe.rebalance_config.dynamic_max_plan_entries_per_wave;
            rebalance_config.dynamic_min_window_activations =
                static_cast<uint32_t>(std::min<uint64_t>(
                    config_.moe.rebalance_config.dynamic_min_window_activations,
                    static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())));
            rebalance_config.routed_assignment_policy =
                config_.moe.routed_assignment_policy == RoutedExpertAssignmentPolicy::LeastLoadedResident
                    ? kDeviceMoERebalanceAssignmentLeastLoadedResident
                    : kDeviceMoERebalanceAssignmentStaticOwner;
            rebalance_config.min_load_spread_improvement = deviceRebalanceConfigOrEnv(
                config_.moe.rebalance_config.device_min_load_spread_improvement,
                "LLAMINAR_MOE_DEVICE_REBALANCE_MIN_LOAD_SPREAD_IMPROVEMENT",
                env.moe_rebalance.device_rebalance_min_load_spread_improvement);
            rebalance_config.min_load_spread_improvement_divisor = deviceRebalanceConfigOrEnv(
                config_.moe.rebalance_config.device_min_load_spread_improvement_divisor,
                "LLAMINAR_MOE_DEVICE_REBALANCE_MIN_LOAD_SPREAD_IMPROVEMENT_DIVISOR",
                env.moe_rebalance.device_rebalance_min_load_spread_improvement_divisor);
            rebalance_config.min_wave_spread_improvement_per_payload_slot =
                deviceRebalanceConfigOrEnv(
                    config_.moe.rebalance_config.device_min_wave_spread_improvement_per_payload_slot,
                    "LLAMINAR_MOE_DEVICE_REBALANCE_MIN_WAVE_SPREAD_IMPROVEMENT_PER_PAYLOAD_SLOT",
                    env.moe_rebalance.device_rebalance_min_wave_spread_improvement_per_payload_slot);
            rebalance_config.min_foreign_rows_per_transfer =
                deviceRebalanceConfigOrEnv(
                    config_.moe.rebalance_config.device_min_foreign_rows_per_transfer,
                    "LLAMINAR_MOE_DEVICE_REBALANCE_MIN_FOREIGN_ROWS_PER_TRANSFER",
                    env.moe_rebalance.device_rebalance_min_foreign_rows_per_transfer);
            rebalance_config.min_router_spread_improvement_per_payload_slot =
                deviceRebalanceConfigOrEnv(
                    config_.moe.rebalance_config.device_min_router_spread_improvement_per_payload_slot,
                    "LLAMINAR_MOE_DEVICE_REBALANCE_MIN_ROUTER_SPREAD_IMPROVEMENT_PER_PAYLOAD_SLOT",
                    env.moe_rebalance.device_rebalance_min_router_spread_improvement_per_payload_slot);
            rebalance_config.max_post_wave_load_spread_per_mille = deviceRebalanceConfigOrEnv(
                config_.moe.rebalance_config.device_max_post_wave_load_spread_per_mille,
                "LLAMINAR_MOE_DEVICE_REBALANCE_MAX_POST_WAVE_LOAD_SPREAD_PERMILLE",
                env.moe_rebalance.device_rebalance_max_post_wave_load_spread_per_mille);
            rebalance_config.llep_alpha_numerator =
                std::max<uint32_t>(1u, config_.moe.rebalance_config.device_llep_alpha_numerator);
            rebalance_config.llep_alpha_denominator =
                std::max<uint32_t>(1u, config_.moe.rebalance_config.device_llep_alpha_denominator);
            rebalance_config.llep_lambda_numerator =
                std::max<uint32_t>(1u, config_.moe.rebalance_config.device_llep_lambda_numerator);
            rebalance_config.llep_lambda_denominator =
                std::max<uint32_t>(1u, config_.moe.rebalance_config.device_llep_lambda_denominator);
            rebalance_config.llep_enable_balanced_skip =
                config_.moe.rebalance_config.device_llep_enable_balanced_skip ? 1u : 0u;
            rebalance_config.flags =
                static_cast<uint32_t>(DeviceMoERebalanceFlags::ResetHistogramsAfterApply);
            if (rebalance_config.max_hot_replicas_per_participant > 0)
            {
                rebalance_config.flags |=
                    static_cast<uint32_t>(DeviceMoERebalanceFlags::HotReplicaCache);
            }
            if (env.moe_rebalance.device_rebalance_maintenance_graph)
            {
                rebalance_config.flags |=
                    static_cast<uint32_t>(DeviceMoERebalanceFlags::DeferRuntimeApply);
            }
            if (env.moe_rebalance.device_rebalance_collect_load_stats ||
                PerfStatsCollector::isEnabled())
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
                const bool dynamic_ownership_transfers =
                    config_.moe.rebalance_mode == MoERebalanceMode::DYNAMIC;
                const bool routed_assignment_payload_transfers =
                    config_.moe.routed_assignment_policy == RoutedExpertAssignmentPolicy::LeastLoadedResident;
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
                uint32_t transfer_slot_count,
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
                auto directory = DeviceMoETransferSlotDirectory::create(
                    backend,
                    device,
                    gpu_ordinal,
                    rebalance_config.participant_id,
                    transfer_slot_count,
                    *graph_rebalance_transfer_profile,
                    gpuDirectRebalanceVramSafetyMarginBytes());
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
            const auto existing = moe_graph_rebalance_bindings_.find(binding_key);
            if (existing != moe_graph_rebalance_bindings_.end())
                return &existing->second;

            DeviceMoERebalanceConfig rebalance_config = makeGraphRebalanceConfig();
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

            const uint64_t persistent_active_slots =
                DeviceMoETransferSlotDirectory::persistentActiveSlotDemand(
                    rebalance_config);
            const uint64_t prefill_llep_slots =
                prefill_llep_transfer_candidate
                    ? static_cast<uint64_t>(
                          std::max(1, env.moe_rebalance.device_rebalance_compact_payload_slots))
                    : 1ULL;
            const uint64_t requested_transfer_slots =
                std::max<uint64_t>(
                    persistent_active_slots,
                    prefill_llep_slots);
            const auto transfer_capacity =
                DeviceMoETransferSlotDirectory::planBufferedCapacity(
                    requested_transfer_slots,
                    static_cast<uint32_t>(
                        std::max(1, env.moe_rebalance.gpu_direct_transfer_wave_experts)),
                    static_cast<uint32_t>(
                        std::max(1, env.moe_rebalance.gpu_direct_transfer_buffers)));
            rebalance_config.active_transfer_slot_capacity =
                transfer_capacity.active_slots;
            rebalance_config.transfer_slot_directory_capacity =
                transfer_capacity.total_slots;
            const uint32_t transfer_slot_count = transfer_capacity.total_slots;
            auto [transfer_key, transfer_directory] =
                getOrCreateGraphRebalanceTransferDirectory(
                    rebalance_config,
                    transfer_slot_count,
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
                static_cast<uint64_t>(
                    ((transfer_directory->wirePayloadBytes() +
                      sizeof(DeviceMoEExpertDirectoryEntry) + 255u) /
                     256u) *
                    256u);

            moe_graph_rebalance_bindings_[binding_key] = GraphSideRebalanceBinding{
                transfer_key,
                rebalance_workspace,
                GraphSideRebalanceBindingRole::PrefillLLEPTransfer,
                device,
                local_tp_ctx,
                local_tp_ctx,
                moe_runtime_table,
                config_.tp_device_idx,
                rebalance_config,
                transfer_directory->deviceEntries(),
                transfer_directory->slotCount(),
                graph_rebalance_transfer_mode.value(),
                collective_payload_slot_bytes,
                collective_payload_slot_capacity,
                state_ref,
                layer_idx};

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
            expert_params.prefill_llep_tp_ctx = binding->decode_tp_ctx;
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
            expert_params.prefill_llep_rebalance_config =
                binding->config;
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
        const bool graph_rebalance_can_have_allreduce_anchor =
            config_.tp_ctx && config_.tp_ctx->degree() > 1;
        auto makeGraphRebalanceStateSidebands =
            [&](const GraphSideRebalanceBinding &binding)
            -> std::vector<TPAllreduceSidebandWorkspaceBinding>
        {
            std::vector<TPAllreduceSidebandWorkspaceBinding> sidebands;
            if (!binding.decode_tp_ctx ||
                !binding.decode_tp_ctx->supportsCollectiveSidebandOnStreamGraphCapture())
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
            params.tp_ctx = binding.decode_tp_ctx;
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
            if (!binding.decode_tp_ctx ||
                !binding.decode_tp_ctx->supportsCollectiveSidebandOnStreamGraphCapture() ||
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

            DeviceMoERebalanceConfig rebalance_config = makeGraphRebalanceConfig();
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

                    const uint64_t requested_transfer_slots =
                        DeviceMoETransferSlotDirectory::
                            persistentActiveSlotDemand(rebalance_config);
                    const auto transfer_capacity =
                        DeviceMoETransferSlotDirectory::planBufferedCapacity(
                            requested_transfer_slots,
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
                    const uint32_t transfer_slot_count =
                        transfer_capacity.total_slots;
                    auto directory_result =
                        getOrCreateGraphRebalanceTransferDirectory(
                            rebalance_config,
                            transfer_slot_count,
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
                        static_cast<uint64_t>(
                            ((transfer_directory->wirePayloadBytes() +
                              sizeof(DeviceMoEExpertDirectoryEntry) + 255u) /
                             256u) *
                            256u);
                }

                ILocalTPContext *maintenance_tp_ctx = local_tp_ctx;
                if (env.moe_rebalance.device_rebalance_maintenance_graph)
                {
                    const std::string maintenance_lane_key =
                        graphRebalanceCollectiveKey();
                    maintenance_tp_ctx =
                        maintenanceTPContextForDomain(maintenance_lane_key, *local_tp_ctx);
                }

                moe_graph_rebalance_bindings_[domain_key] = GraphSideRebalanceBinding{
                    transfer_key,
                    rebalance_workspace,
                    GraphSideRebalanceBindingRole::DecodeMaintenance,
                    device,
                    local_tp_ctx,
                    maintenance_tp_ctx,
                    moe_runtime_table,
                    config_.tp_device_idx,
                    rebalance_config,
                    local_transfer_slots,
                    local_transfer_slot_count,
                    graph_rebalance_transfer_mode.value(),
                    collective_payload_slot_bytes,
                    collective_payload_slot_capacity,
                    transfer_state,
                    layer_idx};
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
            route_params.absolute_position_ids_device =
                device.is_gpu()
                    ? absolute_position_ids_device
                    : nullptr;
            route_params.routed_pipeline_kernel_owner =
                routed_pipeline_kernel_owner;
            route_params.force_decode_equivalent_verifier_prefill =
                forceDecodeEquivalentMoERouting(device);
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
                        if (last_device_rebalance_decode_layer)
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
        bool shared_gate_writes_combined_output = false;
        std::string shared_ffn_last; // Track last shared expert stage (empty if no shared expert)

        auto plannedSharedExpertDevice = [&]() -> DeviceId
        {
            DeviceId shared_device = device;
            if (overlay_runtime_plan)
            {
                const auto &shared_domain = overlay_runtime_plan->sharedExpertDomain();
                shared_device = overlay_runtime_plan->sharedExpertDeviceForMVP(layer_idx);
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
        const RoutedExpertTier *local_tp_replicated_tier = nullptr;
        const bool local_tp_replicated_fast_candidate =
            use_expert_overlay &&
            canUseLocalTPReplicatedFastPath(
                *overlay_plan,
                device,
                &local_tp_replicated_tier);
        const bool ordinary_prefill_graph =
            total_tokens > 1 &&
            !mtp_sidecar_context &&
            !config_.compute_all_position_logits;
        const bool phase_split_local_tp_apportioned_gpu_prefill =
            local_tp_replicated_fast_candidate &&
            local_tp_replicated_tier &&
            ordinary_prefill_graph &&
            isPrefillApportionedDecodeReplicatedTier(
                *overlay_plan,
                *local_tp_replicated_tier);

        auto needsMoEParticipantAllreduce = [&]() -> bool
        {
            return config_.tp_ctx && config_.tp_ctx->degree() > 1 &&
                   (config_.moe.routed_compute_policy !=
                        RoutedExpertComputePolicy::Replicated ||
                    phase_split_local_tp_apportioned_gpu_prefill);
        };
        /*
         * LocalTP expert ownership must never shape the FP32 route addition
         * tree. GPU apportioned paths therefore publish every original route
         * into an independent slot, reduce those slots in FP32 to one fixed
         * participant, fold them in router order on that device, and broadcast
         * only the compact routed row. The shared expert retains its normal
         * allreduce-then-gate arithmetic and is combined only after the routed
         * result has been published to every participant.
         */
        bool canonical_local_tp_route_publication = false;

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
                expert_params.runtime_decode_uses_mutable_descriptors =
                    graphRebalanceDecodeUsesMutableDescriptors() ||
                    activeRuntimeBankUsesTransientLocalPayload(layer_idx);
                expert_params.runtime_decode_has_explicit_owner_metadata =
                    masked_local_tp_overlay_decode_runtime_table ||
                    full_local_tp_replicated_overlay_decode_runtime_table ||
                    masked_local_tp_apportioned_decode_runtime_table;
                expert_params.force_grouped_verifier_prefill_for_decode =
                    forceGroupedMoEVerifierPrefill(stage_device);
                expert_params.defer_grouped_verifier_histogram_publication =
                    forceGpuSmallMMainVerifierPrefill(stage_device);
                expert_params.force_decode_equivalent_verifier_prefill =
                    forceDecodeEquivalentMoEVerifier(stage_device);
                expert_params.absolute_position_ids_device =
                    stage_device.is_gpu()
                        ? absolute_position_ids_device
                        : nullptr;
                expert_params.my_socket_id = std::max(0, config_.tp_device_idx);
                expert_params.participant_count =
                    local_tp_ctx && local_tp_ctx->degree() > 0
                        ? local_tp_ctx->degree()
                        : std::max(1, expert_params.my_socket_id + 1);
                expert_params.routed_assignment_policy =
                    prefill_routed_expert_assignment_policy;
                if (local_tp_ctx &&
                    total_tokens > 1 &&
                    prefill_routed_expert_assignment_policy == RoutedExpertAssignmentPolicy::LeastLoadedResident)
                {
                    expert_params.prefill_llep_tp_ctx = local_tp_ctx;
                    expert_params.prefill_llep_assignment_mode =
                        current_batch_llep_transfer_candidate
                            ? PrefillLLEPAssignmentMode::
                                  TransferBackedCurrentBatch
                            : PrefillLLEPAssignmentMode::
                                  LogicalPositionResidentOnly;
                }

                if (config_.moe.routed_compute_policy == RoutedExpertComputePolicy::Apportioned)
                {
                    expert_params.local_expert_start = config_.moe.local_expert_start;
                    expert_params.local_expert_count = config_.moe.local_expert_count;
                    if (expert_params.local_expert_count >= 0 &&
                        expert_params.expert_mask.empty())
                    {
                        // LocalTP expert-id-apportioned runners own a contiguous
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

                        LOG_TRACE("[Qwen35MoEGraph] Layer " << layer_idx
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

            if (overlay_requested && !use_expert_overlay)
            {
                throw std::runtime_error(
                    "Qwen35 MoE expert overlay was requested but no usable placement exists for layer " +
                    std::to_string(layer_idx) +
                    "; refusing to lower the request through the non-overlay routed path");
            }

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
                    prefill_routed_expert_assignment_policy == RoutedExpertAssignmentPolicy::LeastLoadedResident)
                {
                    expert_params.prefill_llep_tp_ctx = local_tp_ctx;
                    expert_params.prefill_llep_assignment_mode =
                        current_batch_llep_transfer_candidate
                            ? PrefillLLEPAssignmentMode::
                                  TransferBackedCurrentBatch
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
                    expert_params.use_runtime_prefill_grouping = true;
                }

                if (prefill_llep_transfer_candidate &&
                    (expert_params.use_runtime_prefill_grouping ||
                     prefix_runtime_device_rehydration))
                {
                    attachPrefillLLEPTransferBinding(
                        expert_params,
                        "LocalTP expert-ID-apportioned LLEP grouped prefill");
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
                        expert_params.use_runtime_prefill_grouping = true;
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
                        expert_params.use_runtime_prefill_grouping = true;
                    }
                }

                canonical_local_tp_route_publication =
                    needsMoEParticipantAllreduce() &&
                    needsTPAllreduce() &&
                    device.is_gpu();
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
                    expert_params.canonical_route_contributions_buffer_id =
                        buffers.idFor(
                            BufferId::MOE_CANONICAL_ROUTE_CONTRIBUTIONS);
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
                             ? static_cast<size_t>(config_.moe.top_k)
                             : size_t{1});
                    const std::string ar_name =
                        canonical_local_tp_route_publication
                            ? prefix + "moe_canonical_routes_reduce_to_root"
                            : prefix + "moe_expert_overlay_fast_allreduce";
                    auto rebalance_sidebands =
                        takeGraphRebalanceSidebandsForAllreduce();
                    std::unique_ptr<IComputeStage> collective_stage;
                    int canonical_route_root_participant = -1;
                    if (canonical_local_tp_route_publication)
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
                     * The canonical route collective consumes the expert
                     * kernel's per-route publication directly.  Keep that
                     * producer edge explicit even when rebalance state is
                     * piggybacked on the same collective; the collect-state
                     * node is an additional producer, not a substitute for
                     * the tensor producer.  This makes it structurally
                     * impossible for later rebalance graph changes to let the
                     * collective race ahead of the route contribution write.
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

                    if (canonical_local_tp_route_publication)
                    {
                        MoECanonicalRouteReduceStage::Params reduce_params;
                        reduce_params.device_id = device;
                        reduce_params.canonical_route_contributions =
                            canonical_route_contributions;
                        reduce_params.output = moe_output;
                        reduce_params.seq_len = total_tokens;
                        reduce_params.top_k = config_.moe.top_k;
                        reduce_params.d_model = config_.d_model;
                        reduce_params.participant_device_index =
                            config_.tp_device_idx;
                        reduce_params.root_device_index =
                            canonical_route_root_participant;
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
                }

                LOG_TRACE("[Qwen35MoEGraph] Layer " << layer_idx
                                                    << " using LocalTP routed-expert fast path on "
                                                    << device.to_string()
                                                    << " participant=" << local_participant
                                                    << " domain=" << domain_name
                                                    << " compute="
                                                    << routedExpertComputePolicyToString(
                                                           config_.moe.routed_compute_policy)
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
                    const bool local_tp_apportioned_tier =
                        isLocalTPExpertIdApportionedTier(*overlay_plan, tier);
                    const int graph_local_participant =
                        local_tp_apportioned_tier
                            ? participantIdForTierDevice(
                                  *owner_map_lifetime,
                                  static_cast<int>(tier_index),
                                  device)
                            : -1;
                    const bool compute_apportioned_tier_on_graph_local_participant =
                        graph_local_participant >= 0;
                    if (compute_apportioned_tier_on_graph_local_participant)
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
                        if (!MoELocalExpertStage::prepareExpertGemmEngines(local_params))
                        {
                            throw std::runtime_error(
                                "Qwen35 MoE graph failed to prepare participant-local "
                                "expert engines for layer " +
                                std::to_string(layer_idx) + " participant " +
                                std::to_string(target_participant) + " on " +
                                target_device.to_string());
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
                                    graph.addDependency(ar_name, root_return_node);
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

                if (last_return_reduce.empty())
                {
                    throw std::runtime_error("Qwen35 MoE graph-native overlay produced no local expert participants for layer " +
                                             std::to_string(layer_idx));
                }
                ffn_terminal = last_return_reduce;
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
                                  TransferBackedCurrentBatch
                            : PrefillLLEPAssignmentMode::
                                  LogicalPositionResidentOnly;
                }
                if (!prepareExpertParams(expert_params, device))
                {
                    throw std::runtime_error(
                        "Qwen35 MoE graph failed to prepare expert parameters for layer " +
                        std::to_string(layer_idx) + " on " + device.to_string());
                }
                if (prefill_llep_transfer_candidate &&
                    (expert_params.use_runtime_prefill_grouping ||
                     prefix_runtime_device_rehydration))
                {
                    attachPrefillLLEPTransferBinding(
                        expert_params,
                        "standard routed expert LLEP grouped prefill");
                }

                if (device.is_gpu() &&
                    total_tokens == 1 &&
                    moe_runtime_table &&
                    debugEnv().rocm.moe_grouped_decode &&
                    debugEnv().rocm.moe_device_routed_decode)
                {
                    if (masked_local_tp_apportioned_decode_runtime_table)
                    {
                        const int participant_count =
                            local_tp_ctx ? local_tp_ctx->degree() : expert_params.participant_count;
                        const auto owner_participants =
                            contiguousApportionedExpertOwners(
                                config_.moe.num_experts,
                                participant_count);
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
                }

                graph.addNode(prefix + "moe_expert_ffn",
                              ComputeStageFactory::createMoEExpertCompute(expert_params),
                              device);
                const std::string rebalance_apply_dependency =
                    maybeInsertGraphSideRebalance(
                        "standard routed expert path",
                        prefix + "moe_expert_ffn");
                graph.addDependency(prefix + "moe_expert_ffn",
                                    rebalance_apply_dependency.empty()
                                        ? prefix + "moe_routing"
                                        : rebalance_apply_dependency);
                ffn_terminal = prefix + "moe_expert_ffn";

                // Qwen35 MoE expert weights are normally replicated, so every rank
                // computes the full routed-expert contribution. Only allreduce this
                // path when an explicit expert range/mask makes the output partial.
                const bool routed_expert_output_is_partial =
                    expert_params.local_expert_count >= 0 || !expert_params.expert_mask.empty();
                if (routed_expert_output_is_partial && needsTPAllreduce())
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

        if (!shared_gate_writes_combined_output &&
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
            const bool shared_expert_requires_tp_allreduce =
                needsTPAllreduce() && denseTPAllreduceEnabledForCurrentGraph();
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
                mtp_sidecar_context && config_.dense_tp_decode_replicated;
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
            if (shared_gpu_table_verifier_prefill)
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
                 * ordering.  This edge applies equally to main-verifier and MTP
                 * sidecar graphs, and remains valid during whole-graph capture.
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
                 shared_params.force_decode_equivalent_verifier_prefill);
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

            /*
             * Input-parallel prefill shared-expert down rows are reduced before
             * the replicated sigmoid gate, preserving the serial LocalTP branch
             * arithmetic independently of routed-expert placement. Replicated
             * decode rows are already complete and must never be summed again.
             */
            if (shared_expert_requires_tp_allreduce)
            {
                size_t allreduce_count = static_cast<size_t>(total_tokens) * static_cast<size_t>(config_.d_model);
                std::string ar_name = prefix + "shared_expert_allreduce";
                std::vector<TPAllreduceSidebandWorkspaceBinding> rebalance_sidebands;
                if (shared_device == device)
                    rebalance_sidebands = takeGraphRebalanceSidebandsForAllreduce();
                auto allreduce_stage = createTPAllreduceStage(
                    shared_output, allreduce_count, shared_device, layer_idx,
                    /*is_attention=*/false, ar_name, buffers.idFor(BufferId::MOE_SHARED_EXPERT_OUTPUT),
                    std::move(rebalance_sidebands));
                if (allreduce_stage)
                {
                    graph.addNode(ar_name, std::move(allreduce_stage), shared_device);
                    graph.addDependency(ar_name, prefix + "shared_expert_ffn");
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
            if (layer.shared_expert_gate_inp)
            {
                const bool can_fuse_gate_and_combine =
                    !shared_expert_requires_tp_allreduce &&
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
                    // The fused epilogue consumes both the shared-expert output
                    // and the routed-expert output.
                    graph.addDependency(prefix + "shared_expert_gate", ffn_terminal);
                    shared_gate_writes_combined_output = true;
                }
                shared_ffn_last = prefix + "shared_expert_gate";
                if (shared_gate_writes_combined_output)
                {
                    ffn_terminal = prefix + "shared_expert_gate";
                }
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

            auto add_terminal_row_checkpoint =
                [&](const std::string &boundary,
                    const ITensor *source,
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
                checkpoint_params.seq_len = total_tokens;
                checkpoint_params.d_model = config_.d_model;
                checkpoint_params.selected_row_idx = total_tokens - 1;
                configureMirroredCheckpointRowOwnership(
                    checkpoint_params,
                    total_tokens,
                    device,
                    sequence_lengths_device);

                graph.addNode(
                    node_name,
                    ComputeStageFactory::createHiddenStateRowSelect(
                        checkpoint_params),
                    device);
                graph.addDependency(node_name, dependency);
                return node_name;
            };

            const std::string hidden_checkpoint =
                add_terminal_row_checkpoint(
                    "attention_residual",
                    buffers.current_hidden,
                    ffn_terminal);
            ffn_terminal =
                add_terminal_row_checkpoint(
                    "ffn_delta",
                    buffers.attn_proj,
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

    std::string Qwen35MoEGraph::maybeAddGDNDiagnosticCheckpoint(
        ComputeGraph &graph,
        const std::string &boundary,
        const ITensor *source,
        const std::string &dependency,
        int layer_idx,
        int total_tokens,
        int feature_dim,
        DeviceId device,
        const int32_t *sequence_lengths_device)
    {
        if (mtp_graph_context_active_ ||
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
        if (graph_regime == "prefill" &&
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
        const std::string &dependency,
        int total_tokens,
        DeviceId device,
        const int32_t *sequence_lengths_device)
    {
        if (mtp_graph_context_active_ ||
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
         * Exact grouped-verifier, request-condition, and serial-decode graphs
         * have no padding, so their final physical row is also their final
         * logical row. Ordinary GPU prefill may reuse a larger captured bucket;
         * its checkpoint must therefore read the persistent device request
         * length. Refusing to build a prefill checkpoint without that pointer
         * makes selection of a padded row structurally impossible instead of
         * relying on a caller to remember whether this particular shape was
         * exact.
         */
        const bool ordinary_prefill =
            device.is_gpu() &&
            total_tokens > 1 &&
            !config_.grouped_mtp_verifier &&
            !config_.live_mtp_request_batch_condition;
        if (!ordinary_prefill)
        {
            return HiddenStateRowSelectStage::SelectionPolicy::
                FixedDeviceRow;
        }
        if (!sequence_lengths_device)
        {
            throw std::runtime_error(
                "Qwen35 MoE mirrored prefill checkpoint requires a "
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
