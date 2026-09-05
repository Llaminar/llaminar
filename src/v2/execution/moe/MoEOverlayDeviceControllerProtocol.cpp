/**
 * @file MoEOverlayDeviceControllerProtocol.cpp
 * @brief Atomic CPU oracle for the topology-wide GPU controller lifecycle.
 *
 * Stores that publish a transaction word use release ordering after payload
 * metadata. Readers acquire the transaction word before consuming metadata.
 * This is the arithmetic/lifecycle specification mirrored by CUDA and HIP
 * system-scope kernels; it is not a production host policy implementation.
 */

#include "MoEOverlayDeviceControllerProtocol.h"

#include <atomic>
#include <limits>
#include <stdexcept>

namespace llaminar2
{
    namespace
    {
        /** Acquire one shared scalar without changing its plain ABI layout. */
        template <typename T>
        T loadAcquire(T &value) noexcept
        {
            return std::atomic_ref<T>(value).load(std::memory_order_acquire);
        }

        /** Release one shared scalar without embedding `std::atomic` in ABI. */
        template <typename T>
        void storeRelease(T &target, T value) noexcept
        {
            std::atomic_ref<T>(target).store(value, std::memory_order_release);
        }

        /** Return whether @p kind names one supported authority operation. */
        bool validKind(MoEOverlayDeviceControllerTransactionKind kind) noexcept
        {
            return kind ==
                       MoEOverlayDeviceControllerTransactionKind::StaticCheck ||
                   kind == MoEOverlayDeviceControllerTransactionKind::
                               DynamicPlacement ||
                   kind == MoEOverlayDeviceControllerTransactionKind::
                               CurrentBatchLLEP ||
                   kind == MoEOverlayDeviceControllerTransactionKind::
                               PreparedContextRestore;
        }

        /** Return whether @p kind advances one durable placement epoch. */
        bool durablePlacementKind(
            MoEOverlayDeviceControllerTransactionKind kind) noexcept
        {
            return kind == MoEOverlayDeviceControllerTransactionKind::
                               DynamicPlacement ||
                   kind == MoEOverlayDeviceControllerTransactionKind::
                               PreparedContextRestore;
        }
    } // namespace

    void MoEOverlayDeviceControllerProtocol::initialize(
        MoEOverlayDeviceControllerSharedHeader &header,
        std::span<MoEOverlayDeviceControllerGroupRecord> groups,
        MoEOverlayDeviceControllerCommandHeader &command,
        std::uint64_t topology_fingerprint,
        std::uint32_t leader_group_id,
        std::uint32_t leader_participant_id,
        std::uint64_t initial_durable_epoch,
        std::span<const std::uint32_t> group_root_participants)
    {
        if (groups.empty() || groups.size() != group_root_participants.size() ||
            groups.size() > std::numeric_limits<std::uint32_t>::max() ||
            leader_group_id >= groups.size() || topology_fingerprint == 0u ||
            initial_durable_epoch == 0u)
        {
            throw std::invalid_argument(
                "Device controller initialization requires complete non-zero topology identity");
        }
        if (loadAcquire(header.state) !=
            static_cast<std::uint32_t>(
                MoEOverlayDeviceControllerState::Uninitialized))
        {
            throw std::logic_error(
                "Device controller shared header was already initialized");
        }

        header = MoEOverlayDeviceControllerSharedHeader{};
        header.group_count = static_cast<std::uint32_t>(groups.size());
        header.leader_group_id = leader_group_id;
        header.leader_participant_id = leader_participant_id;
        header.topology_fingerprint = topology_fingerprint;
        header.current_durable_epoch = initial_durable_epoch;
        header.admission_epoch = initial_durable_epoch;
        for (std::size_t index = 0; index < groups.size(); ++index)
        {
            groups[index] = MoEOverlayDeviceControllerGroupRecord{};
            groups[index].group_id = static_cast<std::uint32_t>(index);
            groups[index].root_participant_id =
                group_root_participants[index];
            groups[index].topology_fingerprint = topology_fingerprint;
        }
        command = MoEOverlayDeviceControllerCommandHeader{};
        command.topology_fingerprint = topology_fingerprint;
        storeRelease(
            header.state,
            static_cast<std::uint32_t>(
                MoEOverlayDeviceControllerState::Idle));
    }

    MoEOverlayDeviceControllerProtocol::MoEOverlayDeviceControllerProtocol(
        MoEOverlayDeviceControllerSharedHeader &header,
        std::span<MoEOverlayDeviceControllerGroupRecord> groups,
        MoEOverlayDeviceControllerCommandHeader &command) noexcept
        : header_(&header), groups_(groups), command_(&command)
    {
    }

    std::optional<std::uint64_t>
    MoEOverlayDeviceControllerProtocol::beginTransaction(
        MoEOverlayDeviceControllerTransactionKind kind,
        MoEOverlayDeviceDemandPhase phase) noexcept
    {
        const bool valid_phase =
            kind == MoEOverlayDeviceControllerTransactionKind::
                        DynamicPlacement
                ? phase == MoEOverlayDeviceDemandPhase::Prefill ||
                      phase == MoEOverlayDeviceDemandPhase::Decode
                : phase == MoEOverlayDeviceDemandPhase::Invalid;
        if (!valid() || !validKind(kind) || !valid_phase)
        {
            fail(MoEOverlayDeviceControllerError::InvalidControl);
            return std::nullopt;
        }
        const auto current_state = state();
        if (current_state != MoEOverlayDeviceControllerState::Idle &&
            current_state != MoEOverlayDeviceControllerState::Complete)
        {
            fail(MoEOverlayDeviceControllerError::InvalidState);
            return std::nullopt;
        }
        const std::uint64_t previous = loadAcquire(header_->transaction_id);
        if (previous == std::numeric_limits<std::uint64_t>::max())
        {
            fail(MoEOverlayDeviceControllerError::EpochOverflow);
            return std::nullopt;
        }
        const std::uint64_t transaction = previous + 1u;
        const std::uint64_t base = loadAcquire(header_->current_durable_epoch);
        if (base == 0u ||
            (durablePlacementKind(kind) &&
             base == std::numeric_limits<std::uint64_t>::max()))
        {
            fail(MoEOverlayDeviceControllerError::EpochOverflow);
            return std::nullopt;
        }
        const std::uint64_t candidate =
            durablePlacementKind(kind) ? base + 1u : base;

        // Group records are group-root-owned monotonic publications. The
        // leader starts a new transaction without clearing follower words;
        // each root advances its own fields from an older transaction id.
        *command_ = MoEOverlayDeviceControllerCommandHeader{};
        command_->topology_fingerprint = header_->topology_fingerprint;
        header_->transaction_kind = static_cast<std::uint32_t>(kind);
        command_->demand_phase = static_cast<std::uint32_t>(phase);
        header_->base_epoch = base;
        header_->candidate_epoch = candidate;
        header_->command_transaction = 0u;
        header_->commit_transaction = 0u;
        header_->admission_transaction = 0u;
        header_->active_llep_transaction = 0u;
        storeRelease(header_->transaction_id, transaction);
        storeRelease(
            header_->state,
            static_cast<std::uint32_t>(
                MoEOverlayDeviceControllerState::CollectingSnapshots));
        return transaction;
    }

    bool MoEOverlayDeviceControllerProtocol::publishGroupSnapshot(
        std::uint32_t group_id,
        std::uint64_t transaction_id,
        std::uint64_t snapshot_digest,
        std::uint64_t observations) noexcept
    {
        auto *group = requireGroup(group_id);
        if (!group ||
            state() !=
                MoEOverlayDeviceControllerState::CollectingSnapshots ||
            transaction_id == 0u || snapshot_digest == 0u ||
            loadAcquire(header_->transaction_id) != transaction_id)
        {
            return fail(
                MoEOverlayDeviceControllerError::InvalidTransaction,
                group_id);
        }
        const std::uint64_t published =
            loadAcquire(group->snapshot_transaction);
        if (published == transaction_id)
        {
            return group->snapshot_digest == snapshot_digest &&
                   group->snapshot_observations == observations;
        }
        if (published > transaction_id)
        {
            return fail(
                MoEOverlayDeviceControllerError::ConflictingPublication,
                group_id);
        }
        group->snapshot_digest = snapshot_digest;
        group->snapshot_observations = observations;
        storeRelease(group->snapshot_transaction, transaction_id);
        return true;
    }

    bool MoEOverlayDeviceControllerProtocol::allSnapshotsReady(
        std::uint64_t transaction_id) const noexcept
    {
        return valid() && transaction_id != 0u &&
               loadAcquire(header_->transaction_id) == transaction_id &&
               allGroupsMatch(
                   &MoEOverlayDeviceControllerGroupRecord::snapshot_transaction,
                   transaction_id);
    }

    bool MoEOverlayDeviceControllerProtocol::publishCommand(
        std::uint64_t transaction_id,
        std::uint32_t command_count,
        std::uint64_t command_digest,
        std::uint64_t packed_weight_bytes,
        std::uint32_t parallel_command_count,
        std::uint32_t movement_round_count,
        std::uint32_t hazard_count,
        const MoEOverlayDeviceControllerCommandEvidence &evidence) noexcept
    {
        if (state() !=
                MoEOverlayDeviceControllerState::CollectingSnapshots ||
            !allSnapshotsReady(transaction_id) || command_digest == 0u)
        {
            return fail(MoEOverlayDeviceControllerError::InvalidState);
        }
        const auto kind = static_cast<
            MoEOverlayDeviceControllerTransactionKind>(
            header_->transaction_kind);
        const bool has_commands = command_count != 0u;
        const bool empty_shape = !has_commands && packed_weight_bytes == 0u &&
            parallel_command_count == 0u && movement_round_count == 0u &&
            hazard_count == 0u;
        const bool parallel_shape = has_commands &&
            parallel_command_count == command_count &&
            movement_round_count == 1u && hazard_count == 0u;
        const bool command_shape_valid =
            kind == MoEOverlayDeviceControllerTransactionKind::StaticCheck
                ? empty_shape
                : durablePlacementKind(kind)
                ? (empty_shape || (parallel_shape && packed_weight_bytes != 0u))
                : kind == MoEOverlayDeviceControllerTransactionKind::
                              CurrentBatchLLEP &&
                      parallel_shape;
        if (!command_shape_valid)
        {
            return fail(MoEOverlayDeviceControllerError::InvalidCommand);
        }

        // Durable no-op commands are positive certification receipts, but
        // they must not manufacture an epoch that has no physical placement.
        if (durablePlacementKind(kind) && !has_commands)
            header_->candidate_epoch = header_->base_epoch;

        command_->kind = static_cast<std::uint32_t>(kind);
        command_->command_count = command_count;
        command_->topology_fingerprint = header_->topology_fingerprint;
        command_->transaction_id = transaction_id;
        command_->base_epoch = header_->base_epoch;
        command_->candidate_epoch = header_->candidate_epoch;
        command_->command_digest = command_digest;
        command_->packed_weight_bytes = packed_weight_bytes;
        command_->parallel_command_count = parallel_command_count;
        command_->movement_round_count = movement_round_count;
        command_->hazard_count = hazard_count;
        command_->snapshot_observations = evidence.snapshot_observations;
        command_->priority_cost_before = evidence.priority_cost_before;
        command_->priority_cost_after = evidence.priority_cost_after;
        command_->same_priority_makespan_before =
            evidence.same_priority_makespan_before;
        command_->same_priority_makespan_after =
            evidence.same_priority_makespan_after;
        command_->accepted_cycles = evidence.accepted_cycles;
        command_->rejected_cycles = evidence.rejected_cycles;
        command_->promotions = evidence.promotions;
        command_->demotions = evidence.demotions;
        command_->same_priority_moves = evidence.same_priority_moves;
        command_->changed_layers = evidence.changed_layers;
        command_->layer_scan_start = evidence.layer_scan_start;
        command_->layer_scan_next = evidence.layer_scan_next;
        command_->projected_service_gain_ns =
            evidence.projected_service_gain_ns;
        command_->projected_transfer_and_repack_ns =
            evidence.projected_transfer_and_repack_ns;
        command_->projected_inference_interference_ns =
            evidence.projected_inference_interference_ns;
        command_->projected_net_benefit_ns =
            evidence.projected_net_benefit_ns;
        command_->payoff_rejected_cycles =
            evidence.payoff_rejected_cycles;
        command_->residency_rejected_cycles =
            evidence.residency_rejected_cycles;
        storeRelease(header_->command_transaction, transaction_id);
        storeRelease(
            header_->state,
            static_cast<std::uint32_t>(
                MoEOverlayDeviceControllerState::PreparingFollowers));
        return true;
    }

    bool MoEOverlayDeviceControllerProtocol::acknowledgePrepared(
        std::uint32_t group_id,
        std::uint64_t transaction_id) noexcept
    {
        auto *group = requireGroup(group_id);
        if (!group ||
            state() !=
                MoEOverlayDeviceControllerState::PreparingFollowers ||
            loadAcquire(header_->command_transaction) != transaction_id)
        {
            return fail(
                MoEOverlayDeviceControllerError::InvalidTransaction,
                group_id);
        }
        const auto prior = loadAcquire(group->prepared_transaction);
        if (prior > transaction_id)
        {
            return fail(
                MoEOverlayDeviceControllerError::ConflictingPublication,
                group_id);
        }
        storeRelease(group->prepared_transaction, transaction_id);
        return true;
    }

    bool MoEOverlayDeviceControllerProtocol::allGroupsPrepared(
        std::uint64_t transaction_id) const noexcept
    {
        return allGroupsMatch(
            &MoEOverlayDeviceControllerGroupRecord::prepared_transaction,
            transaction_id);
    }

    bool MoEOverlayDeviceControllerProtocol::beginCommit(
        std::uint64_t transaction_id) noexcept
    {
        if (state() !=
                MoEOverlayDeviceControllerState::PreparingFollowers ||
            loadAcquire(header_->command_transaction) != transaction_id ||
            !allGroupsPrepared(transaction_id))
        {
            return fail(MoEOverlayDeviceControllerError::InvalidState);
        }
        storeRelease(header_->commit_transaction, transaction_id);
        storeRelease(
            header_->state,
            static_cast<std::uint32_t>(
                MoEOverlayDeviceControllerState::PublishingFollowers));
        return true;
    }

    bool MoEOverlayDeviceControllerProtocol::acknowledgePublished(
        std::uint32_t group_id,
        std::uint64_t transaction_id) noexcept
    {
        auto *group = requireGroup(group_id);
        if (!group ||
            state() !=
                MoEOverlayDeviceControllerState::PublishingFollowers ||
            loadAcquire(header_->commit_transaction) != transaction_id)
        {
            return fail(
                MoEOverlayDeviceControllerError::InvalidTransaction,
                group_id);
        }
        const auto prior = loadAcquire(group->published_transaction);
        if (prior > transaction_id)
        {
            return fail(
                MoEOverlayDeviceControllerError::ConflictingPublication,
                group_id);
        }
        storeRelease(group->published_transaction, transaction_id);
        return true;
    }

    bool MoEOverlayDeviceControllerProtocol::allGroupsPublished(
        std::uint64_t transaction_id) const noexcept
    {
        return allGroupsMatch(
            &MoEOverlayDeviceControllerGroupRecord::published_transaction,
            transaction_id);
    }

    bool MoEOverlayDeviceControllerProtocol::publishAdmission(
        std::uint64_t transaction_id) noexcept
    {
        if (state() !=
                MoEOverlayDeviceControllerState::PublishingFollowers ||
            !allGroupsPublished(transaction_id) ||
            loadAcquire(header_->commit_transaction) != transaction_id ||
            command_->transaction_id != transaction_id ||
            command_->topology_fingerprint !=
                header_->topology_fingerprint)
        {
            return fail(MoEOverlayDeviceControllerError::InvalidState);
        }
        const auto kind = static_cast<
            MoEOverlayDeviceControllerTransactionKind>(
            header_->transaction_kind);
        if (durablePlacementKind(kind))
        {
            storeRelease(
                header_->current_durable_epoch,
                header_->candidate_epoch);
            storeRelease(header_->admission_epoch, header_->candidate_epoch);
        }
        else
        {
            storeRelease(header_->admission_epoch, header_->base_epoch);
        }
        if (kind ==
            MoEOverlayDeviceControllerTransactionKind::CurrentBatchLLEP)
        {
            storeRelease(header_->active_llep_transaction, transaction_id);
        }
        storeRelease(header_->admission_transaction, transaction_id);
        if (kind ==
            MoEOverlayDeviceControllerTransactionKind::StaticCheck)
        {
            // Publish the durable terminal receipt before exposing Complete.
            // A later transaction may overwrite state and admission fields,
            // but it must never erase proof that this transaction finished.
            storeRelease(header_->completed_transaction, transaction_id);
        }
        storeRelease(
            header_->state,
            static_cast<std::uint32_t>(
                kind == MoEOverlayDeviceControllerTransactionKind::StaticCheck
                    ? MoEOverlayDeviceControllerState::Complete
                    : MoEOverlayDeviceControllerState::Admitted));
        return true;
    }

    bool MoEOverlayDeviceControllerProtocol::beginDynamicRetirement(
        std::uint64_t transaction_id) noexcept
    {
        if (state() != MoEOverlayDeviceControllerState::Admitted ||
            !durablePlacementKind(static_cast<
                MoEOverlayDeviceControllerTransactionKind>(
                header_->transaction_kind)) ||
            loadAcquire(header_->admission_transaction) != transaction_id)
        {
            return fail(MoEOverlayDeviceControllerError::InvalidState);
        }
        storeRelease(
            header_->state,
            static_cast<std::uint32_t>(
                MoEOverlayDeviceControllerState::RetiringDurableEpoch));
        return true;
    }

    bool MoEOverlayDeviceControllerProtocol::acknowledgeRetired(
        std::uint32_t group_id,
        std::uint64_t transaction_id,
        std::uint64_t retired_epoch) noexcept
    {
        auto *group = requireGroup(group_id);
        if (!group ||
            state() !=
                MoEOverlayDeviceControllerState::RetiringDurableEpoch ||
            loadAcquire(header_->admission_transaction) != transaction_id ||
            retired_epoch != header_->base_epoch)
        {
            return fail(
                MoEOverlayDeviceControllerError::InvalidTransaction,
                group_id);
        }
        storeRelease(group->retired_epoch, retired_epoch);
        return true;
    }

    bool MoEOverlayDeviceControllerProtocol::completeDynamicRetirement(
        std::uint64_t transaction_id) noexcept
    {
        if (state() !=
                MoEOverlayDeviceControllerState::RetiringDurableEpoch ||
            loadAcquire(header_->admission_transaction) != transaction_id ||
            !allGroupsMatch(
                &MoEOverlayDeviceControllerGroupRecord::retired_epoch,
                header_->base_epoch))
        {
            return fail(MoEOverlayDeviceControllerError::InvalidState);
        }
        storeRelease(header_->completed_transaction, transaction_id);
        storeRelease(
            header_->state,
            static_cast<std::uint32_t>(
                MoEOverlayDeviceControllerState::Complete));
        return true;
    }

    bool MoEOverlayDeviceControllerProtocol::beginLLEPRestore(
        std::uint64_t transaction_id) noexcept
    {
        if (state() != MoEOverlayDeviceControllerState::Admitted ||
            header_->transaction_kind != static_cast<std::uint32_t>(
                MoEOverlayDeviceControllerTransactionKind::CurrentBatchLLEP) ||
            loadAcquire(header_->active_llep_transaction) != transaction_id)
        {
            return fail(MoEOverlayDeviceControllerError::InvalidState);
        }
        storeRelease(
            header_->state,
            static_cast<std::uint32_t>(
                MoEOverlayDeviceControllerState::RestoringLLEP));
        return true;
    }

    bool MoEOverlayDeviceControllerProtocol::acknowledgeLLEPRestored(
        std::uint32_t group_id,
        std::uint64_t transaction_id) noexcept
    {
        auto *group = requireGroup(group_id);
        if (!group ||
            state() != MoEOverlayDeviceControllerState::RestoringLLEP ||
            loadAcquire(header_->active_llep_transaction) != transaction_id)
        {
            return fail(
                MoEOverlayDeviceControllerError::InvalidTransaction,
                group_id);
        }
        storeRelease(group->restored_transaction, transaction_id);
        return true;
    }

    bool MoEOverlayDeviceControllerProtocol::completeLLEPRestore(
        std::uint64_t transaction_id) noexcept
    {
        if (state() != MoEOverlayDeviceControllerState::RestoringLLEP ||
            !allGroupsMatch(
                &MoEOverlayDeviceControllerGroupRecord::restored_transaction,
                transaction_id) ||
            loadAcquire(header_->active_llep_transaction) != transaction_id)
        {
            return fail(MoEOverlayDeviceControllerError::InvalidState);
        }
        storeRelease(header_->active_llep_transaction, std::uint64_t{0});
        storeRelease(header_->completed_transaction, transaction_id);
        storeRelease(
            header_->state,
            static_cast<std::uint32_t>(
                MoEOverlayDeviceControllerState::Complete));
        return true;
    }

    MoEOverlayDeviceControllerState
    MoEOverlayDeviceControllerProtocol::state() const noexcept
    {
        if (!header_)
            return MoEOverlayDeviceControllerState::Error;
        return static_cast<MoEOverlayDeviceControllerState>(
            loadAcquire(header_->state));
    }

    MoEOverlayDeviceControllerError
    MoEOverlayDeviceControllerProtocol::error() const noexcept
    {
        if (!header_)
            return MoEOverlayDeviceControllerError::InvalidControl;
        return static_cast<MoEOverlayDeviceControllerError>(
            loadAcquire(header_->error_code));
    }

    std::uint64_t
    MoEOverlayDeviceControllerProtocol::currentDurableEpoch() const noexcept
    {
        return header_ ? loadAcquire(header_->current_durable_epoch) : 0u;
    }

    std::uint64_t
    MoEOverlayDeviceControllerProtocol::activeLLEPTransaction() const noexcept
    {
        return header_ ? loadAcquire(header_->active_llep_transaction) : 0u;
    }

    bool MoEOverlayDeviceControllerProtocol::valid() const noexcept
    {
        if (!header_ || !command_ || groups_.empty() ||
            header_->magic != kMoEOverlayDeviceControllerMagic ||
            header_->version != kMoEOverlayDeviceControllerVersion ||
            header_->group_count != groups_.size() ||
            header_->leader_group_id >= groups_.size() ||
            header_->topology_fingerprint == 0u)
        {
            return false;
        }
        for (std::size_t index = 0; index < groups_.size(); ++index)
        {
            const auto &group = groups_[index];
            if (group.magic != kMoEOverlayDeviceControllerMagic ||
                group.version != kMoEOverlayDeviceControllerVersion ||
                group.group_id != index ||
                group.topology_fingerprint != header_->topology_fingerprint)
            {
                return false;
            }
        }
        return true;
    }

    bool MoEOverlayDeviceControllerProtocol::fail(
        MoEOverlayDeviceControllerError error,
        std::uint32_t group_id) noexcept
    {
        if (!header_)
            return false;
        std::uint32_t expected = static_cast<std::uint32_t>(
            MoEOverlayDeviceControllerError::None);
        if (std::atomic_ref<std::uint32_t>(header_->error_code)
                .compare_exchange_strong(
                    expected,
                    static_cast<std::uint32_t>(error),
                    std::memory_order_acq_rel,
                    std::memory_order_acquire))
        {
            header_->error_group_id = group_id;
        }
        storeRelease(
            header_->state,
            static_cast<std::uint32_t>(
                MoEOverlayDeviceControllerState::Error));
        return false;
    }

    MoEOverlayDeviceControllerGroupRecord *
    MoEOverlayDeviceControllerProtocol::requireGroup(
        std::uint32_t group_id) noexcept
    {
        if (!valid() || group_id >= groups_.size())
        {
            fail(MoEOverlayDeviceControllerError::InvalidGroup, group_id);
            return nullptr;
        }
        return &groups_[group_id];
    }

    bool MoEOverlayDeviceControllerProtocol::allGroupsMatch(
        std::uint64_t MoEOverlayDeviceControllerGroupRecord::*field,
        std::uint64_t expected) const noexcept
    {
        if (!valid() || expected == 0u)
            return false;
        for (auto &group : groups_)
        {
            if (loadAcquire(group.*field) != expected)
                return false;
        }
        return true;
    }
} // namespace llaminar2
