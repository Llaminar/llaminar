/**
 * @file MoEOverlayInferenceTransaction.cpp
 * @brief Validation and fixed-ring lifecycle for ExpertOverlay transactions.
 *
 * Every validation is intentionally performed before a slot becomes visible to
 * an MPI request or retained graph. The implementation never infers graph role
 * from row count: one-row prefill, serial decode, MTP sidecar, and grouped
 * verifier are distinct typed roles even when some physical geometries coincide.
 */

#include "MoEOverlayInferenceTransaction.h"
#include "MoEOverlayInferenceOriginalSlot.h"
#include "MoEExpertOwnerMap.h"
#include "execution/mtp/MTPWeightManifest.h"
#include "loaders/IModelLoader.h"

#include <algorithm>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace llaminar2
{
    namespace
    {
        /** @brief Pointer-independent two-lane digest for model topology fields. */
        class StableTopologyDigest final
        {
        public:
            /** @brief Mix one integral or enum value in little-endian order. */
            template <typename Value>
            void addScalar(Value value) noexcept
            {
                if constexpr (std::is_same_v<std::remove_cv_t<Value>, bool>)
                {
                    addByte(value ? 1u : 0u);
                }
                else
                {
                    using Raw = typename std::conditional_t<
                        std::is_enum_v<Value>,
                        std::underlying_type<Value>,
                        std::type_identity<Value>>::type;
                    using Unsigned = std::make_unsigned_t<Raw>;
                    Unsigned bits = static_cast<Unsigned>(value);
                    for (std::size_t index = 0;
                         index < sizeof(Unsigned);
                         ++index)
                    {
                        addByte(static_cast<std::uint8_t>(bits & 0xffu));
                        bits >>= 8u;
                    }
                }
            }

            /** @brief Mix a length-prefixed string without concatenation aliases. */
            void addString(const std::string &value) noexcept
            {
                addScalar(static_cast<std::uint64_t>(value.size()));
                for (const unsigned char byte : value)
                    addByte(byte);
            }

            /** @return Non-zero independent digest lanes. */
            [[nodiscard]] std::pair<std::uint64_t, std::uint64_t>
            finish() const noexcept
            {
                return {
                    low_ == 0 ? 0x9e3779b97f4a7c15ull : low_,
                    high_ == 0 ? 0xd6e8feb86659fd93ull : high_,
                };
            }

        private:
            /** @brief Feed one canonical byte through both independent lanes. */
            void addByte(std::uint8_t byte) noexcept
            {
                low_ ^= static_cast<std::uint64_t>(byte);
                low_ *= 1099511628211ull;
                high_ ^=
                    static_cast<std::uint64_t>(byte) +
                    0x9e3779b97f4a7c15ull;
                high_ *= 14029467366897019727ull;
                high_ ^= high_ >> 29u;
            }

            std::uint64_t low_ = 14695981039346656037ull;
            std::uint64_t high_ = 7809847782465536322ull;
        };

        /** @return Whether @p action is one serialized protocol value. */
        bool validAction(MoEOverlayInferenceTransactionAction action) noexcept
        {
            return action == MoEOverlayInferenceTransactionAction::Execute ||
                   action == MoEOverlayInferenceTransactionAction::Complete ||
                   action == MoEOverlayInferenceTransactionAction::Abort;
        }

        /** @return Whether @p role names an executable retained graph. */
        bool executableRole(MoEOverlayInferenceGraphRole role) noexcept
        {
            return role == MoEOverlayInferenceGraphRole::MainPrefill ||
                   role == MoEOverlayInferenceGraphRole::MainDecode ||
                   role == MoEOverlayInferenceGraphRole::OrdinaryForwardOnly ||
                   role == MoEOverlayInferenceGraphRole::GenerationAdmission ||
                   role == MoEOverlayInferenceGraphRole::MTPDraft ||
                   role == MoEOverlayInferenceGraphRole::MTPGroupedVerifier ||
                   role == MoEOverlayInferenceGraphRole::MTPConditionSpeculativeContinuation ||
                   role == MoEOverlayInferenceGraphRole::MTPConditionCommittedSerialToken;
        }

        /** @return Stable diagnostic name for one ticket action. */
        const char *actionName(
            MoEOverlayInferenceTransactionAction action) noexcept
        {
            switch (action)
            {
            case MoEOverlayInferenceTransactionAction::Execute:
                return "execute";
            case MoEOverlayInferenceTransactionAction::Complete:
                return "complete";
            case MoEOverlayInferenceTransactionAction::Abort:
                return "abort";
            }
            return "invalid";
        }

        /** @return Stable diagnostic name for one retained graph family. */
        const char *roleName(MoEOverlayInferenceGraphRole role) noexcept
        {
            switch (role)
            {
            case MoEOverlayInferenceGraphRole::None:
                return "none";
            case MoEOverlayInferenceGraphRole::MainPrefill:
                return "main_prefill";
            case MoEOverlayInferenceGraphRole::MainDecode:
                return "main_decode";
            case MoEOverlayInferenceGraphRole::OrdinaryForwardOnly:
                return "ordinary_forward_only";
            case MoEOverlayInferenceGraphRole::GenerationAdmission:
                return "generation_admission";
            case MoEOverlayInferenceGraphRole::MTPDraft:
                return "mtp_draft";
            case MoEOverlayInferenceGraphRole::MTPGroupedVerifier:
                return "mtp_grouped_verifier";
            case MoEOverlayInferenceGraphRole::MTPConditionSpeculativeContinuation:
                return "mtp_condition_speculative_continuation";
            case MoEOverlayInferenceGraphRole::MTPConditionCommittedSerialToken:
                return "mtp_condition_committed_serial_token";
            }
            return "invalid";
        }

        /** @brief Store a diagnostic only when the caller requested one. */
        bool fail(std::string message, std::string *error)
        {
            if (error)
                *error = std::move(message);
            return false;
        }
    } // namespace

    bool MoEOverlayInferenceTopologyIdentity::valid() const noexcept
    {
        return workspace_generation != 0 &&
               topology_fingerprint_low != 0 &&
               topology_fingerprint_high != 0 && source_world_rank >= 0 &&
               target_world_rank >= 0;
    }

    bool MoEOverlayInferenceGraphFamilyIdentity::valid() const noexcept
    {
        if (graph_family_generation == 0 || first_model_layer < 0 || main_layer_count <= 0 ||
            main_layer_count > std::numeric_limits<int>::max() - first_model_layer ||
            max_graph_rows <= 0 || max_decode_rows <= 0 ||
            max_decode_rows > max_graph_rows || max_request_count <= 0 ||
            max_mtp_draft_depth < 0)
        {
            return false;
        }
        int previous_layer = first_model_layer + main_layer_count - 1;
        for (const int layer : mtp_source_layers)
        {
            if (layer <= previous_layer ||
                layer == std::numeric_limits<int>::max())
                return false;
            previous_layer = layer;
        }
        return routedLayerCapacity() > 0;
    }

    int MoEOverlayInferenceGraphFamilyIdentity::routedLayerCapacity()
        const noexcept
    {
        if (first_model_layer < 0 || main_layer_count <= 0 ||
            main_layer_count > std::numeric_limits<int>::max() - first_model_layer)
            return 0;
        if (mtp_source_layers.empty())
            return main_layer_count;

        /* Source layers remain model-global wire and prepared-bank indices.
         * A dense NextN source may sit between routed sources, so vector size
         * is not a valid substitute for the final addressable index. */
        const int final_source_layer = mtp_source_layers.back();
        if (final_source_layer < first_model_layer + main_layer_count ||
            final_source_layer == std::numeric_limits<int>::max())
        {
            return 0;
        }
        return final_source_layer - first_model_layer + 1;
    }

    MoEOverlayInferenceGraphRole moeOverlayPipelineConditionGraphRole(MTPConditionForwardPurpose purpose)
    {
        switch (purpose)
        {
        case MTPConditionForwardPurpose::SpeculativeContinuation:
            return MoEOverlayInferenceGraphRole::MTPConditionSpeculativeContinuation;
        case MTPConditionForwardPurpose::CommittedSerialToken:
            return MoEOverlayInferenceGraphRole::MTPConditionCommittedSerialToken;
        }
        throw std::invalid_argument("Pipeline condition ticket has an unknown original purpose");
    }

    MTPConditionForwardPurpose moeOverlayPipelineConditionForwardPurpose(MoEOverlayInferenceGraphRole role)
    {
        switch (role)
        {
        case MoEOverlayInferenceGraphRole::MTPConditionSpeculativeContinuation:
            return MTPConditionForwardPurpose::SpeculativeContinuation;
        case MoEOverlayInferenceGraphRole::MTPConditionCommittedSerialToken:
            return MTPConditionForwardPurpose::CommittedSerialToken;
        default:
            throw std::invalid_argument("Pipeline condition ticket must name an explicit retained purpose");
        }
    }

    MoEOverlayInferenceGraphFamilyIdentity
    MoEOverlayInferenceGraphFamilyIdentity::forRoutedLayerInterval(
        int first, int count) const
    {
        if (!valid() || first < first_model_layer || count <= 0 ||
            first >= first_model_layer + main_layer_count ||
            count > std::numeric_limits<int>::max() - first)
            throw std::invalid_argument("MoE graph family requires an owned nonempty routed stage interval");
        auto stage = *this;
        stage.first_model_layer = first;
        const int remaining_main_layers = first_model_layer + main_layer_count - first;
        stage.main_layer_count = std::min(count, remaining_main_layers);
        if (stage.main_layer_count < remaining_main_layers)
            stage.mtp_source_layers.clear();
        // A terminal stage retains the original manifest, including gaps
        // between routed NextN sources. Nonterminal stages own no sidecars.
        if (!stage.valid() || stage.routedLayerCapacity() != count)
            throw std::invalid_argument("MoE stage placement differs from its retained main/NextN graph family");
        return stage;
    }

    MoEOverlayInferenceGraphFamilyIdentity
    resolveMoEOverlayInferenceGraphFamilyIdentity(
        const IModelLoader &loader,
        const std::string &architecture,
        int raw_layer_count,
        MoEOverlayMTPGraphFamilyPolicy mtp_policy,
        std::uint64_t graph_family_generation,
        int max_graph_rows,
        int max_decode_rows,
        int max_request_count,
        int max_mtp_draft_depth)
    {
        MoEOverlayInferenceGraphFamilyIdentity family;
        family.graph_family_generation = graph_family_generation;
        family.main_layer_count = mainLayerCountExcludingMTP(
            loader, architecture, raw_layer_count);
        family.max_graph_rows = max_graph_rows;
        family.max_decode_rows = max_decode_rows;
        family.max_request_count = max_request_count;
        family.max_mtp_draft_depth = max_mtp_draft_depth;

        if (mtp_policy ==
            MoEOverlayMTPGraphFamilyPolicy::RetainModelSidecars)
        {
            const MTPWeightManifest manifest = discoverMTPWeightManifest(
                loader,
                architecture,
                raw_layer_count,
                /*explicit_mtp=*/true);
            if (!manifest.available)
            {
                throw std::invalid_argument(
                    "MTP-enabled ExpertOverlay transaction family could not resolve its NextN manifest: " +
                    manifest.diagnostic);
            }
            for (const auto &depth : manifest.depths)
            {
                if (!depth.moe_ffn_layout)
                    continue;
                if (depth.depth_index !=
                    static_cast<int>(family.mtp_source_layers.size()))
                {
                    throw std::invalid_argument(
                        "ExpertOverlay transaction family requires contiguous routed MTP graph depths");
                }
                family.mtp_source_layers.push_back(
                    depth.source_layer_index);
            }
        }

        if (raw_layer_count <= 0 ||
            family.main_layer_count > raw_layer_count || !family.valid())
        {
            throw std::invalid_argument(
                "ExpertOverlay transaction family resolved invalid model or capacity geometry");
        }
        return family;
    }

    namespace
    {
    /** @brief Hash the complete validated owner topology and retained family. */
    MoEOverlayInferenceTopologyIdentity makeInferenceTopologyIdentity(
        const MoEExpertOwnerMap &owner_map,
        const MoEOverlayInferenceGraphFamilyIdentity &graph_family,
        int source_world_rank,
        int target_world_rank,
        int local_source_participant = -1,
        std::span<const int> local_targets = {})
    {
        if (!graph_family.valid() || source_world_rank < 0 ||
            target_world_rank < 0 ||
            (source_world_rank == target_world_rank &&
             local_source_participant < 0))
        {
            throw std::invalid_argument(
                "Cannot fingerprint invalid ExpertOverlay inference graph geometry or rank pair");
        }

        std::vector<const MoEExpertOwnerParticipant *> participants;
        participants.reserve(owner_map.participants().size());
        bool source_present = false;
        bool target_present = false;
        for (const auto &participant : owner_map.participants())
        {
            participants.push_back(&participant);
            source_present = source_present ||
                             (participant.world_rank_known &&
                              participant.world_rank == source_world_rank);
            target_present = target_present ||
                             (participant.world_rank_known &&
                              participant.world_rank == target_world_rank);
        }
        if (!source_present || !target_present)
        {
            throw std::invalid_argument(
                "ExpertOverlay inference rank pair is absent from the frozen participant topology");
        }
        std::sort(
            participants.begin(),
            participants.end(),
            [](const auto *left, const auto *right)
            {
                return left->participant_id < right->participant_id;
            });

        StableTopologyDigest digest;
        digest.addString("MoEOverlayInferenceTopology/v2");
        digest.addScalar(source_world_rank);
        digest.addScalar(target_world_rank);
        if (local_source_participant >= 0)
        {
            digest.addString("RankLocalEndpoints/v1");
            digest.addScalar(local_source_participant);
            digest.addScalar(static_cast<std::uint64_t>(local_targets.size()));
            for (const int participant : local_targets)
                digest.addScalar(participant);
        }
        digest.addScalar(graph_family.first_model_layer);
        digest.addScalar(graph_family.main_layer_count);
        digest.addScalar(graph_family.max_graph_rows);
        digest.addScalar(graph_family.max_decode_rows);
        digest.addScalar(graph_family.max_request_count);
        digest.addScalar(graph_family.max_mtp_draft_depth);
        digest.addScalar(static_cast<std::uint64_t>(
            graph_family.mtp_source_layers.size()));
        for (const int layer : graph_family.mtp_source_layers)
            digest.addScalar(layer);

        digest.addScalar(
            static_cast<std::uint64_t>(participants.size()));
        for (const auto *participant : participants)
        {
            digest.addScalar(participant->participant_id);
            digest.addScalar(participant->tier_idx);
            digest.addString(participant->tier_name);
            digest.addString(participant->domain_name);
            digest.addScalar(participant->domain_participant_index);
            digest.addString(participant->address.hostname);
            digest.addScalar(participant->address.numa_node);
            digest.addScalar(participant->address.device_type);
            digest.addScalar(participant->address.device_ordinal);
            digest.addScalar(participant->world_rank);
            digest.addScalar(participant->world_rank_known);
        }
        const auto [low, high] = digest.finish();
        return {
            .workspace_generation =
                graph_family.graph_family_generation,
            .topology_fingerprint_low = low,
            .topology_fingerprint_high = high,
            .source_world_rank = source_world_rank,
            .target_world_rank = target_world_rank,
        };
    }

    } // namespace

    MoEOverlayInferenceTopologyIdentity
    makeMoEOverlayInferenceTopologyIdentity(
        const MoEExpertOwnerMap &owner_map,
        const MoEOverlayInferenceGraphFamilyIdentity &graph_family,
        int source_world_rank,
        int target_world_rank)
    {
        return makeInferenceTopologyIdentity(
            owner_map, graph_family, source_world_rank, target_world_rank);
    }

    MoEOverlayInferenceTopologyIdentity
    makeMoEOverlayRankLocalInferenceTopologyIdentity(
        const MoEExpertOwnerMap &owner_map,
        const MoEOverlayInferenceGraphFamilyIdentity &graph_family,
        int source_participant_id,
        std::span<const int> target_participant_ids)
    {
        const auto *source = owner_map.participantForId(source_participant_id);
        if (!source || !source->world_rank_known ||
            source->world_rank < 0 || !source->device.is_gpu() ||
            target_participant_ids.empty() ||
            !std::is_sorted(target_participant_ids.begin(), target_participant_ids.end()) ||
            std::adjacent_find(target_participant_ids.begin(), target_participant_ids.end()) !=
                target_participant_ids.end())
        {
            throw std::invalid_argument(
                "Rank-local expert endpoints require one original source and ordered unique targets");
        }
        for (const int id : target_participant_ids)
        {
            const auto *target = owner_map.participantForId(id);
            if (!target || !target->world_rank_known ||
                target->world_rank != source->world_rank ||
                target->domain_name == source->domain_name ||
                !target->device.is_gpu() ||
                std::any_of(owner_map.participants().begin(), owner_map.participants().end(),
                    [&](const auto &continuation)
                    {
                        return continuation.domain_name == source->domain_name &&
                               continuation.world_rank_known &&
                               continuation.world_rank == source->world_rank &&
                               continuation.device == target->device;
                    }))
            {
                throw std::invalid_argument(
                    "Rank-local expert endpoint is absent, foreign, or belongs to the continuation domain");
            }
        }
        return makeInferenceTopologyIdentity(
            owner_map, graph_family, source->world_rank, source->world_rank,
            source_participant_id, target_participant_ids);
    }

    bool MoEOverlayInferenceCommandIdentity::valid() const noexcept
    {
        const bool placement_valid =
            (placement_identity_kind == MoEOverlayInferencePlacementIdentityKind::ResidencyEpoch &&
             initial_placement_epoch != 0) ||
            (placement_identity_kind == MoEOverlayInferencePlacementIdentityKind::RetainedGraph &&
             initial_placement_epoch == 0) ||
            placement_identity_kind == MoEOverlayInferencePlacementIdentityKind::ControllerPlacementEpoch;
        return request_generation != 0 && command_id != 0 && placement_valid;
    }

    MoEOverlayInferenceTopologyIdentity
    MoEOverlayInferenceTransactionTicket::topologyIdentity() const noexcept
    {
        return {
            .workspace_generation = workspace_generation,
            .topology_fingerprint_low = topology_fingerprint_low,
            .topology_fingerprint_high = topology_fingerprint_high,
            .source_world_rank = source_world_rank,
            .target_world_rank = target_world_rank,
        };
    }

    MoEOverlayInferenceCommandIdentity
    MoEOverlayInferenceTransactionTicket::commandIdentity() const noexcept
    {
        return {
            .request_generation = request_generation,
            .command_id = command_id,
            .initial_placement_epoch = placement_epoch,
            .placement_identity_kind = placement_identity_kind,
        };
    }

    MoEOverlayInferenceWorkloadIdentity
    MoEOverlayInferenceTransactionTicket::prefillScheduleWorkload()
        const noexcept
    {
        if (prefill_schedule_real_rows == 0 &&
            prefill_schedule_execution_rows == 0 &&
            prefill_schedule_transaction_count == 0 &&
            prefill_schedule_fingerprint == 0)
        {
            return {};
        }
        return {
            .source = ExpertHistogramSource::PrefillChunk,
            .real_rows = prefill_schedule_real_rows,
            .execution_rows = prefill_schedule_execution_rows,
            .transaction_count = prefill_schedule_transaction_count,
            .speculative_depth = 0,
            .schedule_fingerprint = prefill_schedule_fingerprint,
        };
    }

    bool MoEOverlayInferenceTransactionTicket::valid() const noexcept
    {
        const bool no_prefill_schedule =
            prefill_schedule_real_rows == 0 &&
            prefill_schedule_execution_rows == 0 &&
            prefill_schedule_transaction_count == 0 &&
            prefill_schedule_fingerprint == 0;
        const auto prefill_schedule = prefillScheduleWorkload();
        const bool valid_prefill_schedule =
            prefill_schedule.valid() &&
            prefill_schedule.source == ExpertHistogramSource::PrefillChunk &&
            prefill_schedule.speculative_depth == 0;
        if (magic != kMagic || abi_version != kABIVersion ||
            !validAction(action) || !topologyIdentity().valid() ||
            request_generation == 0 || command_id == 0 ||
            transaction_ordinal == 0 || !commandIdentity().valid() ||
            (!no_prefill_schedule && !valid_prefill_schedule))
        {
            return false;
        }

        if (action != MoEOverlayInferenceTransactionAction::Execute)
        {
            const bool valid_terminal_code =
                (action == MoEOverlayInferenceTransactionAction::Complete &&
                 error_code == 0) ||
                (action == MoEOverlayInferenceTransactionAction::Abort &&
                 error_code > 0);
            return valid_terminal_code &&
                   graph_role == MoEOverlayInferenceGraphRole::None &&
                   request_count == 0 && logical_rows_per_request == 0 &&
                   physical_rows_per_request == 0 && draft_depth == -1 &&
                   sidecar_depth == -1 && no_prefill_schedule;
        }

        if (!executableRole(graph_role) || error_code != 0 ||
            request_count <= 0 || logical_rows_per_request <= 0 ||
            physical_rows_per_request < logical_rows_per_request ||
            (!no_prefill_schedule &&
             graph_role != MoEOverlayInferenceGraphRole::MainPrefill))
        {
            return false;
        }

        switch (graph_role)
        {
        case MoEOverlayInferenceGraphRole::MainPrefill:
            return draft_depth == -1 && sidecar_depth == -1;
        case MoEOverlayInferenceGraphRole::MainDecode:
            return logical_rows_per_request == 1 && draft_depth == -1 &&
                   sidecar_depth == -1;
        case MoEOverlayInferenceGraphRole::GenerationAdmission:
        case MoEOverlayInferenceGraphRole::OrdinaryForwardOnly:
        case MoEOverlayInferenceGraphRole::MTPConditionSpeculativeContinuation:
        case MoEOverlayInferenceGraphRole::MTPConditionCommittedSerialToken:
            return logical_rows_per_request == 1 && physical_rows_per_request == 1 &&
                   draft_depth == -1 && sidecar_depth == -1;
        case MoEOverlayInferenceGraphRole::MTPDraft:
            return logical_rows_per_request == 1 && draft_depth > 0 &&
                   sidecar_depth >= 0 && sidecar_depth < draft_depth;
        case MoEOverlayInferenceGraphRole::MTPGroupedVerifier:
            return draft_depth > 0 && sidecar_depth == -1 &&
                   logical_rows_per_request == draft_depth + 1;
        case MoEOverlayInferenceGraphRole::None:
            return false;
        }
        return false;
    }

    std::string MoEOverlayInferenceTransactionTicket::toString() const
    {
        std::ostringstream stream;
        stream << "action=" << actionName(action)
               << ",role=" << roleName(graph_role)
               << ",request_generation=" << request_generation
               << ",command=" << command_id
               << ",transaction=" << transaction_ordinal
               << ",logical_step=" << logical_step_id
               << ",workspace_generation=" << workspace_generation
               << ",placement_authority=" << static_cast<std::uint32_t>(placement_identity_kind)
               << ",placement_epoch=" << placement_epoch
               << ",retired_decode_progress_tokens="
               << retired_decode_progress_tokens
               << ",topology_low=" << topology_fingerprint_low
               << ",topology_high=" << topology_fingerprint_high
               << ",source_rank=" << source_world_rank
               << ",target_rank=" << target_world_rank
               << ",requests=" << request_count
               << ",logical_rows_per_request="
               << logical_rows_per_request
               << ",physical_rows_per_request="
               << physical_rows_per_request
               << ",draft_depth=" << draft_depth
               << ",sidecar_depth=" << sidecar_depth
               << ",error_code=" << error_code
               << ",prefill_schedule_real_rows="
               << prefill_schedule_real_rows
               << ",prefill_schedule_execution_rows="
               << prefill_schedule_execution_rows
               << ",prefill_schedule_transactions="
               << prefill_schedule_transaction_count
               << ",prefill_schedule_fingerprint="
               << prefill_schedule_fingerprint;
        return stream.str();
    }

    MoEOverlayInferenceTransactionTicket
    makeMoEOverlayInferenceExecutionTicket(
        const MoEOverlayInferenceTopologyIdentity &topology,
        const MoEOverlayInferenceCommandIdentity &command,
        std::uint64_t transaction_ordinal,
        std::uint64_t logical_step_id,
        std::uint64_t placement_epoch,
        MoEOverlayInferenceGraphRole graph_role,
        int request_count,
        int logical_rows_per_request,
        int physical_rows_per_request,
        int draft_depth,
        int sidecar_depth,
        const MoEOverlayInferenceWorkloadIdentity &
            prefill_schedule_workload,
        std::uint64_t retired_decode_progress_tokens)
    {
        const bool carries_prefill_schedule =
            prefill_schedule_workload.valid();
        if (carries_prefill_schedule &&
            (graph_role != MoEOverlayInferenceGraphRole::MainPrefill ||
             prefill_schedule_workload.source !=
                 ExpertHistogramSource::PrefillChunk ||
             prefill_schedule_workload.speculative_depth != 0))
        {
            throw std::invalid_argument(
                "ExpertOverlay aggregate prefill workload may accompany only a main-prefill ticket");
        }
        MoEOverlayInferenceTransactionTicket ticket{
            .magic = MoEOverlayInferenceTransactionTicket::kMagic,
            .abi_version =
                MoEOverlayInferenceTransactionTicket::kABIVersion,
            .action = MoEOverlayInferenceTransactionAction::Execute,
            .graph_role = graph_role,
            .request_generation = command.request_generation,
            .command_id = command.command_id,
            .transaction_ordinal = transaction_ordinal,
            .logical_step_id = logical_step_id,
            .workspace_generation = topology.workspace_generation,
            .placement_epoch = placement_epoch,
            .retired_decode_progress_tokens =
                retired_decode_progress_tokens,
            .topology_fingerprint_low = topology.topology_fingerprint_low,
            .topology_fingerprint_high = topology.topology_fingerprint_high,
            .source_world_rank = topology.source_world_rank,
            .target_world_rank = topology.target_world_rank,
            .request_count = request_count,
            .logical_rows_per_request = logical_rows_per_request,
            .physical_rows_per_request = physical_rows_per_request,
            .draft_depth = draft_depth,
            .sidecar_depth = sidecar_depth,
            .error_code = 0,
            .prefill_schedule_real_rows =
                carries_prefill_schedule
                    ? prefill_schedule_workload.real_rows
                    : 0,
            .prefill_schedule_execution_rows =
                carries_prefill_schedule
                    ? prefill_schedule_workload.execution_rows
                    : 0,
            .prefill_schedule_transaction_count =
                carries_prefill_schedule
                    ? prefill_schedule_workload.transaction_count
                    : 0,
            .placement_identity_kind = command.placement_identity_kind,
            .prefill_schedule_fingerprint =
                carries_prefill_schedule
                    ? prefill_schedule_workload.schedule_fingerprint
                    : 0,
        };
        if (!topology.valid() || !command.valid() || !ticket.valid() ||
            placement_epoch < command.initial_placement_epoch)
        {
            throw std::invalid_argument(
                "Invalid ExpertOverlay execution transaction ticket: " +
                ticket.toString());
        }
        return ticket;
    }

    MoEOverlayInferenceTransactionTicket
    makeMoEOverlayInferenceTerminalTicket(
        const MoEOverlayInferenceTopologyIdentity &topology,
        const MoEOverlayInferenceCommandIdentity &command,
        std::uint64_t transaction_ordinal,
        std::uint64_t placement_epoch,
        MoEOverlayInferenceTransactionAction action,
        int error_code,
        std::uint64_t retired_decode_progress_tokens)
    {
        MoEOverlayInferenceTransactionTicket ticket{
            .magic = MoEOverlayInferenceTransactionTicket::kMagic,
            .abi_version =
                MoEOverlayInferenceTransactionTicket::kABIVersion,
            .action = action,
            .graph_role = MoEOverlayInferenceGraphRole::None,
            .request_generation = command.request_generation,
            .command_id = command.command_id,
            .transaction_ordinal = transaction_ordinal,
            .workspace_generation = topology.workspace_generation,
            .placement_epoch = placement_epoch,
            .retired_decode_progress_tokens =
                retired_decode_progress_tokens,
            .topology_fingerprint_low = topology.topology_fingerprint_low,
            .topology_fingerprint_high = topology.topology_fingerprint_high,
            .source_world_rank = topology.source_world_rank,
            .target_world_rank = topology.target_world_rank,
            .request_count = 0,
            .logical_rows_per_request = 0,
            .physical_rows_per_request = 0,
            .draft_depth = -1,
            .sidecar_depth = -1,
            .error_code = error_code,
            .placement_identity_kind = command.placement_identity_kind,
        };
        if (!topology.valid() || !command.valid() || !ticket.valid() ||
            placement_epoch < command.initial_placement_epoch)
        {
            throw std::invalid_argument(
                "Invalid ExpertOverlay terminal transaction ticket: " +
                ticket.toString());
        }
        return ticket;
    }

    MoEOverlayInferenceTransactionProtocol::
        MoEOverlayInferenceTransactionProtocol(Config config)
        : config_(std::move(config)), slots_(config_.slot_count)
    {
        if (!config_.topology.valid() || config_.slot_count == 0 ||
            config_.max_request_count <= 0 ||
            config_.max_rows_per_request <= 0 ||
            config_.max_mtp_draft_depth < 0)
        {
            throw std::invalid_argument(
                "ExpertOverlay inference transaction protocol requires valid topology, positive row/request capacities, and non-negative MTP depth");
        }
        original_slot_domain_ = std::shared_ptr<const MoEOverlayInferenceOriginalSlotDomain>(
            new MoEOverlayInferenceOriginalSlotDomain(this, slots_.size(), config_.max_mtp_draft_depth, config_.topology));
    }

    MoEOverlayInferenceTransactionProtocol::~MoEOverlayInferenceTransactionProtocol()
    {
        if (original_slot_domain_)
        {
            std::lock_guard lock(original_slot_domain_->ownership_mutex_);
            if (original_slot_domain_->owner_ == this)
                original_slot_domain_->owner_ = nullptr;
        }
    }

    const MoEOverlayInferenceTransactionProtocol &
    MoEOverlayInferenceOriginalSlotDomain::requireRetainedOwner(
        const MoEOverlayInferenceOriginalSlot &slot) const
    {
        if (!owner_ || slot.domain().get() != this ||
            slot.domain().owner_before(owner_->original_slot_domain_) ||
            owner_->original_slot_domain_.owner_before(slot.domain()) ||
            slot.index() >= owner_->slots_.size())
            throw std::logic_error("Native resource names a foreign or withdrawn original slot owner");
        const auto &original = owner_->slots_[slot.index()];
        if (original.state == MoEOverlayInferenceTransactionSlotState::Available ||
            original.ticket != slot.ticket())
            throw std::logic_error("Native resource names a retired or replaced original publication slot");
        return *owner_;
    }

    void MoEOverlayInferenceOriginalSlotDomain::requireCurrent(
        const MoEOverlayInferenceOriginalSlot &slot) const
    {
        std::lock_guard lock(ownership_mutex_);
        (void)requireRetainedOwner(slot);
    }

    void MoEOverlayInferenceOriginalSlotDomain::requireActiveOwnerLocked(
        const MoEOverlayInferenceOriginalSlot &slot) const
    {
        const auto &owner = requireRetainedOwner(slot);
        if (owner.state_ != MoEOverlayInferenceProtocolState::Active ||
            owner.slots_[slot.index()].state != MoEOverlayInferenceTransactionSlotState::Submitted)
            throw std::logic_error("Native execution admission lost its original active submitted protocol slot");
    }

    void MoEOverlayInferenceOriginalSlotDomain::requireActiveAdmission(
        const MoEOverlayInferenceOriginalSlot &slot) const
    {
        std::lock_guard lock(ownership_mutex_);
        requireActiveOwnerLocked(slot);
    }

    std::shared_ptr<const MoEOverlayInferenceOriginalSlotDomain>
    MoEOverlayInferenceTransactionProtocol::originalSlotDomain() const
    {
        std::lock_guard lock(original_slot_domain_->ownership_mutex_);
        if (original_slot_domain_->owner_ != this)
            throw std::logic_error("Native invocation slots require their original protocol owner");
        return original_slot_domain_;
    }

    MoEOverlayInferenceOriginalSlot
    MoEOverlayInferenceTransactionProtocol::originalSubmittedSlot(
        const MoEOverlayInferenceTransactionTicket &ticket) const
    {
        const auto domain = originalSlotDomain();
        std::lock_guard lock(domain->ownership_mutex_);
        for (std::size_t index = 0; index < slots_.size(); ++index)
        {
            const auto &slot = slots_[index];
            if (state_ == MoEOverlayInferenceProtocolState::Active &&
                slot.state == MoEOverlayInferenceTransactionSlotState::Submitted &&
                slot.ticket == ticket)
                return MoEOverlayInferenceOriginalSlot(domain, index, ticket);
        }
        throw std::invalid_argument("Native invocation has no exact original submitted protocol slot");
    }

    bool MoEOverlayInferenceTransactionProtocol::beginCommand(
        const MoEOverlayInferenceCommandIdentity &command,
        std::string *error)
    {
        std::lock_guard native_slot_lock(original_slot_domain_->ownership_mutex_);
        if (error)
            error->clear();
        if (!command.valid())
            return fail("ExpertOverlay command identity is invalid", error);
        if (state_ == MoEOverlayInferenceProtocolState::Active)
            return fail("ExpertOverlay command cannot begin while another command is active", error);
        if (state_ == MoEOverlayInferenceProtocolState::Failed)
            return fail("ExpertOverlay failed protocol is process-terminal", error);
        if (inFlightSlotCount() != 0)
            return fail("ExpertOverlay command cannot begin with live transaction slots", error);

        const bool newer_request =
            command.request_generation > last_request_generation_;
        const bool next_command_same_request =
            command.request_generation == last_request_generation_ &&
            command.command_id > last_command_id_;
        if (last_request_generation_ != 0 && !newer_request &&
            !next_command_same_request)
        {
            return fail(
                "ExpertOverlay command generation/id is stale or non-monotonic",
                error);
        }

        active_command_ = command;
        last_request_generation_ = command.request_generation;
        last_command_id_ = command.command_id;
        next_transaction_ordinal_ = 1;
        current_placement_epoch_ = command.initial_placement_epoch;
        state_ = MoEOverlayInferenceProtocolState::Active;
        return true;
    }

    std::string MoEOverlayInferenceTransactionProtocol::
        validateExecutionTicket(
            const MoEOverlayInferenceTransactionTicket &ticket) const
    {
        if (state_ != MoEOverlayInferenceProtocolState::Active)
            return "ExpertOverlay transaction arrived without an active command";
        if (!ticket.valid())
            return "ExpertOverlay transaction ticket has invalid ABI or role geometry";
        if (ticket.topologyIdentity() != config_.topology)
            return "ExpertOverlay transaction topology/workspace identity diverged";
        if (ticket.request_generation != active_command_.request_generation ||
            ticket.command_id != active_command_.command_id)
        {
            return "ExpertOverlay transaction belongs to a stale or future command";
        }
        if (ticket.transaction_ordinal != next_transaction_ordinal_)
            return "ExpertOverlay transaction ordinal is duplicate or out of order";
        if (ticket.placement_identity_kind != active_command_.placement_identity_kind)
            return "ExpertOverlay transaction changed its placement authority";
        if (ticket.placement_epoch < current_placement_epoch_)
            return "ExpertOverlay transaction regressed its placement epoch";
        if (ticket.action == MoEOverlayInferenceTransactionAction::Execute)
        {
            if (ticket.request_count > config_.max_request_count ||
                ticket.logical_rows_per_request >
                    config_.max_rows_per_request ||
                ticket.physical_rows_per_request >
                    config_.max_rows_per_request)
            {
                return "ExpertOverlay transaction exceeds its fixed graph geometry";
            }
            if ((ticket.graph_role ==
                     MoEOverlayInferenceGraphRole::MTPDraft ||
                 ticket.graph_role ==
                     MoEOverlayInferenceGraphRole::MTPGroupedVerifier) &&
                ticket.draft_depth > config_.max_mtp_draft_depth)
            {
                return "ExpertOverlay MTP transaction exceeds admitted maximum depth";
            }
        }
        return {};
    }

    MoEOverlayInferenceAdmission
    MoEOverlayInferenceTransactionProtocol::accept(
        const MoEOverlayInferenceTransactionTicket &ticket)
    {
        std::lock_guard native_slot_lock(original_slot_domain_->ownership_mutex_);
        MoEOverlayInferenceAdmission admission;
        admission.error = validateExecutionTicket(ticket);
        if (!admission.error.empty())
            return admission;

        if (ticket.action == MoEOverlayInferenceTransactionAction::Abort)
        {
            state_ = MoEOverlayInferenceProtocolState::Failed;
            ++next_transaction_ordinal_;
            admission.status = MoEOverlayInferenceAdmissionStatus::Aborted;
            admission.error = "ExpertOverlay command aborted with error code " +
                              std::to_string(ticket.error_code);
            return admission;
        }

        if (ticket.action == MoEOverlayInferenceTransactionAction::Complete)
        {
            if (inFlightSlotCount() != 0)
            {
                admission.status =
                    MoEOverlayInferenceAdmissionStatus::Backpressured;
                admission.error =
                    "ExpertOverlay terminal ticket arrived before every transaction slot retired";
                return admission;
            }
            if (ticket.placement_epoch != current_placement_epoch_)
            {
                admission.error =
                    "ExpertOverlay terminal ticket cannot publish a new placement epoch";
                return admission;
            }
            state_ = MoEOverlayInferenceProtocolState::Complete;
            ++next_transaction_ordinal_;
            admission.status = MoEOverlayInferenceAdmissionStatus::Complete;
            admission.error.clear();
            return admission;
        }

        const std::size_t slot_index = static_cast<std::size_t>(
            (ticket.transaction_ordinal - 1u) % slots_.size());
        Slot &slot = slots_[slot_index];
        if (slot.state != MoEOverlayInferenceTransactionSlotState::Available)
        {
            admission.status =
                MoEOverlayInferenceAdmissionStatus::Backpressured;
            admission.error =
                "ExpertOverlay fixed transaction slot is still owned by prior work";
            return admission;
        }

        // Publish the complete ticket before exposing Accepted to the caller.
        slot.ticket = ticket;
        slot.state = MoEOverlayInferenceTransactionSlotState::Accepted;
        current_placement_epoch_ = ticket.placement_epoch;
        ++next_transaction_ordinal_;
        admission.status = MoEOverlayInferenceAdmissionStatus::Accepted;
        admission.slot_index = slot_index;
        admission.error.clear();
        return admission;
    }

    bool MoEOverlayInferenceTransactionProtocol::transitionSlot(
        std::size_t slot_index,
        const MoEOverlayInferenceTransactionTicket &ticket,
        MoEOverlayInferenceTransactionSlotState expected,
        MoEOverlayInferenceTransactionSlotState next,
        std::string *error)
    {
        std::lock_guard native_slot_lock(original_slot_domain_->ownership_mutex_);
        if (error)
            error->clear();
        if (slot_index >= slots_.size())
            return fail("ExpertOverlay transaction slot index is out of range", error);
        Slot &slot = slots_[slot_index];
        if (slot.state != expected)
            return fail("ExpertOverlay transaction slot lifecycle is out of order", error);
        if (slot.ticket != ticket)
            return fail("ExpertOverlay transaction slot rejected an aliasing ticket", error);

        slot.state = next;
        if (next == MoEOverlayInferenceTransactionSlotState::Available)
            slot.ticket = MoEOverlayInferenceTransactionTicket{};
        return true;
    }

    bool MoEOverlayInferenceTransactionProtocol::markSubmitted(
        std::size_t slot_index,
        const MoEOverlayInferenceTransactionTicket &ticket,
        std::string *error)
    {
        return transitionSlot(
            slot_index,
            ticket,
            MoEOverlayInferenceTransactionSlotState::Accepted,
            MoEOverlayInferenceTransactionSlotState::Submitted,
            error);
    }

    bool MoEOverlayInferenceTransactionProtocol::markReturnReady(
        std::size_t slot_index,
        const MoEOverlayInferenceTransactionTicket &ticket,
        std::string *error)
    {
        return transitionSlot(
            slot_index,
            ticket,
            MoEOverlayInferenceTransactionSlotState::Submitted,
            MoEOverlayInferenceTransactionSlotState::ReturnReady,
            error);
    }

    bool MoEOverlayInferenceTransactionProtocol::retire(
        std::size_t slot_index,
        const MoEOverlayInferenceTransactionTicket &ticket,
        std::string *error)
    {
        return transitionSlot(
            slot_index,
            ticket,
            MoEOverlayInferenceTransactionSlotState::ReturnReady,
            MoEOverlayInferenceTransactionSlotState::Available,
            error);
    }

    std::size_t
    MoEOverlayInferenceTransactionProtocol::inFlightSlotCount() const noexcept
    {
        std::size_t count = 0;
        for (const Slot &slot : slots_)
        {
            if (slot.state !=
                MoEOverlayInferenceTransactionSlotState::Available)
            {
                ++count;
            }
        }
        return count;
    }

    std::optional<MoEOverlayInferenceTransactionSlotState>
    MoEOverlayInferenceTransactionProtocol::slotState(
        std::size_t slot_index) const noexcept
    {
        if (slot_index >= slots_.size())
            return std::nullopt;
        return slots_[slot_index].state;
    }

} // namespace llaminar2
