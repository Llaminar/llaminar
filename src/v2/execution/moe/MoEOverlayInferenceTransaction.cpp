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
#include "MoEExpertOwnerMap.h"
#include "execution/mtp/MTPWeightManifest.h"
#include "loaders/IModelLoader.h"

#include <algorithm>
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
                   role == MoEOverlayInferenceGraphRole::MTPDraft ||
                   role ==
                       MoEOverlayInferenceGraphRole::MTPGroupedVerifier;
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
            case MoEOverlayInferenceGraphRole::MTPDraft:
                return "mtp_draft";
            case MoEOverlayInferenceGraphRole::MTPGroupedVerifier:
                return "mtp_grouped_verifier";
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
               target_world_rank >= 0 &&
               source_world_rank != target_world_rank;
    }

    bool MoEOverlayInferenceGraphFamilyIdentity::valid() const noexcept
    {
        if (graph_family_generation == 0 || main_layer_count <= 0 ||
            max_graph_rows <= 0 || max_decode_rows <= 0 ||
            max_decode_rows > max_graph_rows || max_request_count <= 0 ||
            max_mtp_draft_depth < 0)
        {
            return false;
        }
        int previous_layer = main_layer_count - 1;
        for (const int layer : mtp_source_layers)
        {
            if (layer <= previous_layer)
                return false;
            previous_layer = layer;
        }
        return true;
    }

    MoEOverlayInferenceGraphFamilyIdentity
    resolveMoEOverlayInferenceGraphFamilyIdentity(
        const IModelLoader &loader,
        const std::string &architecture,
        int raw_layer_count,
        bool mtp_enabled,
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

        if (mtp_enabled)
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

    MoEOverlayInferenceTopologyIdentity
    makeMoEOverlayInferenceTopologyIdentity(
        const MoEExpertOwnerMap &owner_map,
        const MoEOverlayInferenceGraphFamilyIdentity &graph_family,
        int source_world_rank,
        int target_world_rank)
    {
        if (!graph_family.valid() || source_world_rank < 0 ||
            target_world_rank < 0 ||
            source_world_rank == target_world_rank)
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
        digest.addString("MoEOverlayInferenceTopology/v1");
        digest.addScalar(source_world_rank);
        digest.addScalar(target_world_rank);
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

    bool MoEOverlayInferenceCommandIdentity::valid() const noexcept
    {
        return request_generation != 0 && command_id != 0 &&
               initial_placement_epoch != 0;
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
        };
    }

    bool MoEOverlayInferenceTransactionTicket::valid() const noexcept
    {
        if (magic != kMagic || abi_version != kABIVersion ||
            !validAction(action) || !topologyIdentity().valid() ||
            request_generation == 0 || command_id == 0 ||
            transaction_ordinal == 0 || placement_epoch == 0)
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
                   sidecar_depth == -1;
        }

        if (!executableRole(graph_role) || error_code != 0 ||
            request_count <= 0 || logical_rows_per_request <= 0 ||
            physical_rows_per_request < logical_rows_per_request)
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
               << ",placement_epoch=" << placement_epoch
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
               << ",error_code=" << error_code;
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
        int sidecar_depth)
    {
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
        int error_code)
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
    }

    bool MoEOverlayInferenceTransactionProtocol::beginCommand(
        const MoEOverlayInferenceCommandIdentity &command,
        std::string *error)
    {
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
