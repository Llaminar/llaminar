/**
 * @file MoEOverlayDeviceTransportProtocol.cpp
 * @brief Host-side immutable-command authentication and physical completion.
 *
 * Physical transport owns only its completion lane. Device-owned participant
 * and group receipts authorize lifecycle progress. Action-entry observations
 * enrich fatal diagnostics but are never read by an admission predicate.
 */

#include "MoEOverlayDeviceTransportProtocol.h"

#include "MoEOverlayDeviceControllerKernels.h"

#include <atomic>
#include <cstring>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace llaminar2
{
    namespace
    {
        /** Acquire a device-published mapped scalar through its plain ABI word. */
        template <typename T>
        T loadAcquire(const T &value) noexcept
        {
            return std::atomic_ref<T>(const_cast<T &>(value)).load(
                std::memory_order_acquire);
        }

        /** Release one host-owned transport scalar to the group-root GPU. */
        template <typename T>
        void storeRelease(T &target, T value) noexcept
        {
            std::atomic_ref<T>(target).store(value, std::memory_order_release);
        }

        /** Copy a fixed-width record after its release publication is acquired. */
        template <typename T>
        T snapshotRecord(const T *source) noexcept
        {
            T result{};
            if (source)
                std::memcpy(&result, source, sizeof(T));
            return result;
        }

        /** Populate an optional error and return false for concise guards. */
        bool reject(std::string *error, std::string message) noexcept
        {
            if (error)
                *error = std::move(message);
            return false;
        }

        /** @return Whether @p kind owns a durable placement epoch. */
        bool durablePlacementKind(
            MoEOverlayDeviceControllerTransactionKind kind) noexcept
        {
            return kind == MoEOverlayDeviceControllerTransactionKind::
                               DynamicPlacement ||
                   kind == MoEOverlayDeviceControllerTransactionKind::
                               PreparedContextRestore;
        }

        /**
         * @return A bounded diagnostic for the sole authority's terminal edge.
         *
         * The transport worker is permitted to observe controller lifecycle
         * identity and its first terminal error, but no placement payload. This
         * turns an already-published device failure into an immediate scheduler
         * failure instead of hiding it behind the ordinary command deadline.
         */
        std::string controllerFailureMessage(
            const MoEOverlayDeviceControllerSharedHeader &controller)
        {
            std::ostringstream out;
            out << "device overlay authority is terminal: state="
                << loadAcquire(controller.state)
                << " error_code=" << loadAcquire(controller.error_code)
                << " error_group_id="
                << loadAcquire(controller.error_group_id)
                << " transaction="
                << loadAcquire(controller.transaction_id)
                << " command_transaction="
                << loadAcquire(controller.command_transaction);
            return out.str();
        }

        /** @return Exact digest over the fixed-width command byte sequence. */
        std::uint64_t commandDigest(
            const std::vector<MoEOverlayDeviceMovementCommand> &entries) noexcept
        {
            std::uint64_t digest = moeOverlayCommandDigestSeed(
                static_cast<std::uint32_t>(entries.size()));
            constexpr std::size_t kWordsPerCommand =
                sizeof(MoEOverlayDeviceMovementCommand) /
                sizeof(std::uint64_t);
            const auto *words = reinterpret_cast<const std::uint64_t *>(
                entries.data());
            const std::size_t word_count = entries.size() * kWordsPerCommand;
            for (std::size_t index = 0u; index < word_count; ++index)
            {
                digest ^= moeOverlayCommandDigestWord(words[index], index);
            }
            return digest;
        }

        /** Participant-owned lifecycle word observed by physical transport. */
        enum class ParticipantPhase
        {
            Snapshot,
            Prepared,
            Published,
            RetirementReady,
            Retired,
            Restored,
        };

        /** @return One participant's monotonic word for @p phase. */
        std::uint64_t participantPhaseWord(
            const MoEOverlayDeviceControllerParticipantRecord &record,
            ParticipantPhase phase) noexcept
        {
            switch (phase)
            {
            case ParticipantPhase::Snapshot:
                return loadAcquire(record.snapshot_transaction);
            case ParticipantPhase::Prepared:
                return loadAcquire(record.prepared_transaction);
            case ParticipantPhase::Published:
                return loadAcquire(record.published_transaction);
            case ParticipantPhase::RetirementReady:
                return loadAcquire(record.retirement_ready_epoch);
            case ParticipantPhase::Retired:
                return loadAcquire(record.retired_epoch);
            case ParticipantPhase::Restored:
                return loadAcquire(record.restored_transaction);
            }
            return 0u;
        }

        /**
         * @return Whether every device in this transport group crossed one RCU edge.
         *
         * These records expose lifecycle only. The physical worker cannot read
         * runtime descriptors, histograms, or placement policy through its
         * binding, so waiting here cannot turn it into a second authority.
         */
        bool allParticipantsReached(
            const MoEOverlayDeviceControllerTransportBinding &binding,
            ParticipantPhase phase,
            std::uint64_t expected) noexcept
        {
            if (!binding.valid() || expected == 0u)
                return false;
            for (std::uint32_t index = 0u;
                 index < binding.participant_record_count;
                 ++index)
            {
                const auto &record = binding.participant_records[index];
                if (record.magic !=
                        kMoEOverlayDeviceControllerFabricMagic ||
                    record.version !=
                        kMoEOverlayDeviceControllerFabricVersion ||
                    record.group_id !=
                        static_cast<std::uint32_t>(binding.group_id) ||
                    record.topology_fingerprint !=
                        binding.topology_fingerprint ||
                    loadAcquire(record.status_code) !=
                        static_cast<std::uint32_t>(
                            MoEOverlayDeviceControllerError::None) ||
                    participantPhaseWord(record, phase) < expected)
                {
                    return false;
                }
            }
            return true;
        }

        /**
         * @return Whether every topology participant crossed one lifecycle edge.
         *
         * Retirement is the only scheduler edge that must be topology-wide:
         * reclaiming any old bank is legal only after no participant can still
         * acquire that epoch. The const grouped aliases expose lifecycle words
         * only and therefore cannot become a host placement authority.
         */
        bool allTopologyParticipantsReached(
            const MoEOverlayDeviceControllerTransportBinding &binding,
            ParticipantPhase phase,
            std::uint64_t expected) noexcept
        {
            if (!binding.valid() || expected == 0u)
                return false;
            for (std::uint32_t group = 0u;
                 group < binding.group_record_count;
                 ++group)
            {
                const auto *records =
                    binding.topology_participant_records[group];
                const auto count =
                    binding.topology_participant_record_counts[group];
                if (!records || count == 0u)
                    return false;
                for (std::uint32_t member = 0u; member < count; ++member)
                {
                    const auto &record = records[member];
                    if (record.magic !=
                            kMoEOverlayDeviceControllerFabricMagic ||
                        record.version !=
                            kMoEOverlayDeviceControllerFabricVersion ||
                        record.group_id != group ||
                        record.topology_fingerprint !=
                            binding.topology_fingerprint ||
                        loadAcquire(record.status_code) !=
                            static_cast<std::uint32_t>(
                                MoEOverlayDeviceControllerError::None) ||
                        participantPhaseWord(record, phase) < expected)
                    {
                        return false;
                    }
                }
            }
            return true;
        }

        /** Group-root lifecycle word observed by the bounded phase scheduler. */
        enum class GroupPhase
        {
            Snapshot,
            Prepared,
            Published,
            Retired,
        };

        /** @return One group root's monotonic word for @p phase. */
        std::uint64_t groupPhaseWord(
            const MoEOverlayDeviceControllerGroupRecord &record,
            GroupPhase phase) noexcept
        {
            switch (phase)
            {
            case GroupPhase::Snapshot:
                return loadAcquire(record.snapshot_transaction);
            case GroupPhase::Prepared:
                return loadAcquire(record.prepared_transaction);
            case GroupPhase::Published:
                return loadAcquire(record.published_transaction);
            case GroupPhase::Retired:
                return loadAcquire(record.retired_epoch);
            }
            return 0u;
        }

        /**
         * @return Whether every topology group crossed one lifecycle edge.
         *
         * The const binding exposes only group acknowledgement records. The
         * worker can schedule the next immutable graph phase, but cannot read
         * collected histograms or author a placement decision.
         */
        bool allGroupsReached(
            const MoEOverlayDeviceControllerTransportBinding &binding,
            GroupPhase phase,
            std::uint64_t expected) noexcept
        {
            if (!binding.valid() || expected == 0u)
                return false;
            for (std::uint32_t index = 0u;
                 index < binding.group_record_count;
                 ++index)
            {
                const auto *record = binding.group_records[index];
                if (!record ||
                    record->magic != kMoEOverlayDeviceControllerMagic ||
                    record->version != kMoEOverlayDeviceControllerVersion ||
                    record->group_id != index ||
                    record->topology_fingerprint !=
                        binding.topology_fingerprint ||
                    loadAcquire(record->status_code) !=
                        static_cast<std::uint32_t>(
                            MoEOverlayDeviceControllerError::None) ||
                    groupPhaseWord(*record, phase) < expected)
                {
                    return false;
                }
            }
            return true;
        }

        /** @return Whether @p batch still names the live controller transaction. */
        bool exactControllerTransaction(
            const MoEOverlayDeviceControllerTransportBinding &binding,
            const MoEOverlayDeviceTransportCommandBatch &batch,
            MoEOverlayDeviceControllerState state) noexcept
        {
            return binding.valid() && batch.valid() &&
                batch.header.topology_fingerprint ==
                    binding.topology_fingerprint &&
                loadAcquire(binding.controller->transaction_id) ==
                    batch.header.transaction_id &&
                loadAcquire(binding.controller->transaction_kind) ==
                    batch.header.kind &&
                loadAcquire(binding.controller->state) ==
                    static_cast<std::uint32_t>(state);
        }

        /**
         * @return Whether a mapped participant or this physical lane is terminal.
         *
         * Participant actions precede group-root fan-in on the same retained
         * graph. During retirement the root intentionally waits for a later
         * host publication edge, so relying only on the global controller
         * state can hide an earlier participant fault until the protocol
         * deadline. These are lifecycle/error words only; observing them does
         * not give the host placement or policy authority.
         */
        bool localProtocolFailurePublished(
            const MoEOverlayDeviceControllerTransportBinding &binding) noexcept
        {
            if (!binding.valid())
                return true;
            if (loadAcquire(binding.transport->status_code) !=
                static_cast<std::uint32_t>(
                    MoEOverlayDeviceControllerError::None))
            {
                return true;
            }
            for (std::uint32_t index = 0u;
                 index < binding.participant_record_count;
                 ++index)
            {
                const auto &record = binding.participant_records[index];
                if (record.magic !=
                        kMoEOverlayDeviceControllerFabricMagic ||
                    record.version !=
                        kMoEOverlayDeviceControllerFabricVersion ||
                    record.group_id !=
                        static_cast<std::uint32_t>(binding.group_id) ||
                    record.topology_fingerprint !=
                        binding.topology_fingerprint ||
                    loadAcquire(record.status_code) !=
                        static_cast<std::uint32_t>(
                            MoEOverlayDeviceControllerError::None))
                {
                    return true;
                }
            }
            return false;
        }

        /** @return Whether any topology group root published a terminal code. */
        bool topologyGroupFailurePublished(
            const MoEOverlayDeviceControllerTransportBinding &binding) noexcept
        {
            if (!binding.valid())
                return true;
            for (std::uint32_t index = 0u;
                 index < binding.group_record_count;
                 ++index)
            {
                const auto *record = binding.group_records[index];
                if (!record ||
                    loadAcquire(record->status_code) !=
                        static_cast<std::uint32_t>(
                            MoEOverlayDeviceControllerError::None))
                {
                    return true;
                }
            }
            return false;
        }
    } // namespace

    bool MoEOverlayDeviceTransportCommandBatch::valid() const noexcept
    {
        if (header.magic != kMoEOverlayDeviceControllerMagic ||
            header.version != kMoEOverlayDeviceControllerVersion ||
            header.topology_fingerprint == 0u || header.transaction_id == 0u ||
            header.base_epoch == 0u || participant_count == 0u ||
            num_layers == 0u || num_experts == 0u ||
            entries.size() != header.command_count ||
            header.parallel_command_count != header.command_count ||
            header.movement_round_count !=
                (header.command_count == 0u ? 0u : 1u) ||
            header.hazard_count != 0u ||
            commandDigest(entries) != header.command_digest)
        {
            return false;
        }

        const auto kind = static_cast<
            MoEOverlayDeviceControllerTransactionKind>(header.kind);
        const auto demand_phase = static_cast<MoEOverlayDeviceDemandPhase>(
            header.demand_phase);
        const bool phase_valid =
            kind == MoEOverlayDeviceControllerTransactionKind::
                        DynamicPlacement
                ? demand_phase == MoEOverlayDeviceDemandPhase::Prefill ||
                      demand_phase == MoEOverlayDeviceDemandPhase::Decode
                : demand_phase == MoEOverlayDeviceDemandPhase::Invalid;
        if (!phase_valid)
            return false;
        std::uint64_t payload_bytes = 0u;
        for (std::size_t index = 0u; index < entries.size(); ++index)
        {
            const auto &entry = entries[index];
            if (!moeOverlayMovementCommandValid(
                    entry,
                    static_cast<std::uint32_t>(index),
                    kind,
                    participant_count,
                    num_layers,
                    num_experts,
                    header.base_epoch,
                    header.candidate_epoch) ||
                (index != 0u &&
                 !moeOverlayIndependentCanonicalDestination(
                     entries[index - 1u], entry)) ||
                entry.payload_bytes >
                    std::numeric_limits<std::uint64_t>::max() - payload_bytes)
            {
                return false;
            }
            payload_bytes += entry.payload_bytes;
        }
        if (payload_bytes != header.packed_weight_bytes)
            return false;

        switch (kind)
        {
        case MoEOverlayDeviceControllerTransactionKind::StaticCheck:
            return entries.empty() && header.packed_weight_bytes == 0u &&
                   header.candidate_epoch == header.base_epoch &&
                   header.accepted_cycles == 0u &&
                   header.promotions == 0u && header.demotions == 0u &&
                   header.same_priority_moves == 0u &&
                   header.changed_layers == 0u &&
                   header.layer_scan_start == 0u &&
                   header.layer_scan_next == 0u &&
                   header.projected_service_gain_ns == 0u &&
                   header.projected_transfer_and_repack_ns == 0u &&
                   header.projected_inference_interference_ns == 0u &&
                   header.projected_net_benefit_ns == 0u &&
                   header.payoff_rejected_cycles == 0u &&
                   header.residency_rejected_cycles == 0u;
        case MoEOverlayDeviceControllerTransactionKind::DynamicPlacement:
        {
            const bool has_movement = !entries.empty();
            const bool measured_cost_fits =
                header.projected_transfer_and_repack_ns <=
                    std::numeric_limits<std::uint64_t>::max() -
                        header.projected_inference_interference_ns;
            const std::uint64_t measured_cost = measured_cost_fits
                ? header.projected_transfer_and_repack_ns +
                      header.projected_inference_interference_ns
                : std::numeric_limits<std::uint64_t>::max();
            const bool measured_economy_is_coherent = has_movement
                ? measured_cost_fits &&
                      header.projected_service_gain_ns > measured_cost &&
                      header.projected_net_benefit_ns ==
                          header.projected_service_gain_ns - measured_cost
                : header.projected_service_gain_ns == 0u &&
                      header.projected_transfer_and_repack_ns == 0u &&
                      header.projected_inference_interference_ns == 0u &&
                      header.projected_net_benefit_ns == 0u;
            return header.candidate_epoch ==
                       (has_movement ? header.base_epoch + 1u
                                     : header.base_epoch) &&
                   (entries.empty() ==
                    (header.packed_weight_bytes == 0u)) &&
                   (has_movement == (header.accepted_cycles != 0u)) &&
                   (has_movement == (header.changed_layers != 0u)) &&
                   static_cast<std::uint64_t>(header.promotions) +
                           header.demotions + header.same_priority_moves ==
                       entries.size() &&
                   header.layer_scan_start < num_layers &&
                   header.layer_scan_next < num_layers &&
                   measured_economy_is_coherent;
        }
        case MoEOverlayDeviceControllerTransactionKind::
            PreparedContextRestore:
        {
            const bool has_movement = !entries.empty();
            return header.candidate_epoch ==
                       (has_movement ? header.base_epoch + 1u
                                     : header.base_epoch) &&
                   (entries.empty() ==
                    (header.packed_weight_bytes == 0u)) &&
                   (has_movement == (header.accepted_cycles != 0u)) &&
                   (has_movement == (header.changed_layers != 0u)) &&
                   static_cast<std::uint64_t>(header.promotions) +
                           header.demotions + header.same_priority_moves ==
                       entries.size() &&
                   header.layer_scan_start < num_layers &&
                   header.layer_scan_next < num_layers &&
                   header.priority_cost_before == 0u &&
                   header.priority_cost_after == 0u &&
                   header.same_priority_makespan_before == 0u &&
                   header.same_priority_makespan_after == 0u &&
                   header.projected_service_gain_ns == 0u &&
                   header.projected_transfer_and_repack_ns == 0u &&
                   header.projected_inference_interference_ns == 0u &&
                   header.projected_net_benefit_ns == 0u &&
                   header.payoff_rejected_cycles == 0u &&
                   header.residency_rejected_cycles == 0u;
        }
        case MoEOverlayDeviceControllerTransactionKind::CurrentBatchLLEP:
            return !entries.empty() &&
                   header.candidate_epoch == header.base_epoch &&
                   header.projected_service_gain_ns == 0u &&
                   header.projected_transfer_and_repack_ns == 0u &&
                   header.projected_inference_interference_ns == 0u &&
                   header.projected_net_benefit_ns == 0u &&
                   header.payoff_rejected_cycles == 0u &&
                   header.residency_rejected_cycles == 0u;
        case MoEOverlayDeviceControllerTransactionKind::Invalid:
            return false;
        }
        return false;
    }

    MoEOverlayDeviceTransportProtocol::MoEOverlayDeviceTransportProtocol(
        MoEOverlayDeviceControllerTransportBinding binding)
        : binding_(std::move(binding))
    {
        if (!binding_.valid())
        {
            throw std::invalid_argument(
                "device transport protocol requires one valid local group binding");
        }
    }

    MoEOverlayDeviceTransportAcquireResult
    MoEOverlayDeviceTransportProtocol::tryAcquire(
        std::uint64_t after_transaction) noexcept
    {
        MoEOverlayDeviceTransportAcquireResult result;
        if (!binding_.valid())
        {
            result.status = MoEOverlayDeviceTransportAcquireStatus::Failed;
            result.error = "device transport binding lost immutable identity";
            fail();
            return result;
        }
        if (loadAcquire(binding_.transport->status_code) !=
            static_cast<std::uint32_t>(
                MoEOverlayDeviceControllerError::None))
        {
            result.status = MoEOverlayDeviceTransportAcquireStatus::Failed;
            result.error = "device transport lane is already terminal";
            return result;
        }

        /* Policy authoring may fail before command_transaction is published.
         * That is a terminal device-owned edge, not an absent command. Observe
         * it before returning Waiting so the scheduler never converts a precise
         * controller failure into a derivative 30-second transport timeout. */
        if (loadAcquire(binding_.controller->state) ==
                static_cast<std::uint32_t>(
                    MoEOverlayDeviceControllerState::Error) ||
            loadAcquire(binding_.controller->error_code) !=
                static_cast<std::uint32_t>(
                    MoEOverlayDeviceControllerError::None))
        {
            result.status = MoEOverlayDeviceTransportAcquireStatus::Failed;
            result.error = controllerFailureMessage(*binding_.controller);
            return result;
        }

        const std::uint64_t published = loadAcquire(
            binding_.controller->command_transaction);
        if (published == 0u || published <= after_transaction)
            return result;

        const auto header = snapshotRecord(binding_.command);
        if (header.command_count > binding_.command_capacity)
        {
            result.status = MoEOverlayDeviceTransportAcquireStatus::Failed;
            result.error = "device command exceeds the pre-materialized transport capacity";
            fail();
            return result;
        }

        result.batch.header = header;
        result.batch.participant_count = binding_.layout->participant_count;
        result.batch.num_layers = binding_.layout->num_layers;
        result.batch.num_experts = binding_.layout->num_experts;
        result.batch.entries.resize(header.command_count);
        if (header.command_count != 0u)
        {
            std::memcpy(
                result.batch.entries.data(),
                binding_.command_entries,
                static_cast<std::size_t>(header.command_count) *
                    sizeof(MoEOverlayDeviceMovementCommand));
        }
        std::atomic_thread_fence(std::memory_order_acquire);
        if (loadAcquire(binding_.controller->command_transaction) != published ||
            header.transaction_id != published ||
            header.topology_fingerprint != binding_.topology_fingerprint ||
            !result.batch.valid())
        {
            result.status = MoEOverlayDeviceTransportAcquireStatus::Failed;
            result.error = "device transport rejected a stale or invalid immutable command batch";
            fail();
            return result;
        }

        auto &record = *binding_.transport;
        record.prepared_transaction = 0u;
        record.published_transaction = 0u;
        record.retired_epoch = 0u;
        record.restored_transaction = 0u;
        record.command_digest = header.command_digest;
        record.status_code = static_cast<std::uint32_t>(
            MoEOverlayDeviceControllerError::None);
        record.state = static_cast<std::uint32_t>(
            MoEOverlayDeviceControllerTransportState::CommandAcquired);
        std::atomic_thread_fence(std::memory_order_release);
        storeRelease(record.command_transaction, published);

        result.status = MoEOverlayDeviceTransportAcquireStatus::Ready;
        return result;
    }

    bool MoEOverlayDeviceTransportProtocol::snapshotTransactionAfter(
        std::uint64_t after_transaction,
        std::uint64_t *transaction,
        MoEOverlayDeviceControllerTransactionKind *kind,
        MoEOverlayDeviceDemandPhase *phase) const noexcept
    {
        if (transaction)
            *transaction = 0u;
        if (kind)
            *kind = MoEOverlayDeviceControllerTransactionKind::Invalid;
        if (phase)
            *phase = MoEOverlayDeviceDemandPhase::Invalid;
        if (!transaction || !kind || !phase || !binding_.valid() ||
            loadAcquire(binding_.controller->state) !=
                static_cast<std::uint32_t>(
                    MoEOverlayDeviceControllerState::CollectingSnapshots))
        {
            return false;
        }
        const auto observed_kind =
            static_cast<MoEOverlayDeviceControllerTransactionKind>(
                loadAcquire(binding_.controller->transaction_kind));
        if (!durablePlacementKind(observed_kind))
            return false;
        const std::uint64_t observed = loadAcquire(
            binding_.controller->transaction_id);
        const auto observed_phase =
            static_cast<MoEOverlayDeviceDemandPhase>(
                loadAcquire(binding_.controller->transaction_demand_phase));
        const bool valid_phase =
            observed_kind == MoEOverlayDeviceControllerTransactionKind::
                                 DynamicPlacement
                ? observed_phase == MoEOverlayDeviceDemandPhase::Prefill ||
                      observed_phase == MoEOverlayDeviceDemandPhase::Decode
                : observed_phase == MoEOverlayDeviceDemandPhase::Invalid;
        if (observed == 0u || observed <= after_transaction || !valid_phase)
            return false;
        *transaction = observed;
        *kind = observed_kind;
        *phase = observed_phase;
        return true;
    }

    bool MoEOverlayDeviceTransportProtocol::localSnapshotsReady(
        std::uint64_t transaction) const noexcept
    {
        return transaction != 0u && binding_.valid() &&
            loadAcquire(binding_.controller->transaction_id) == transaction &&
            loadAcquire(binding_.controller->state) ==
                static_cast<std::uint32_t>(
                    MoEOverlayDeviceControllerState::CollectingSnapshots) &&
            allParticipantsReached(
                binding_, ParticipantPhase::Snapshot, transaction);
    }

    bool MoEOverlayDeviceTransportProtocol::allGroupsSnapshotted(
        std::uint64_t transaction) const noexcept
    {
        // Group records are durable scheduler receipts. The authority may have
        // advanced after observing the last group but before a remote rank's
        // host scheduler polls this predicate.
        return transaction != 0u && binding_.valid() &&
            loadAcquire(binding_.controller->transaction_id) >= transaction &&
            allGroupsReached(binding_, GroupPhase::Snapshot, transaction);
    }

    bool MoEOverlayDeviceTransportProtocol::requireAcquired(
        const MoEOverlayDeviceTransportCommandBatch &batch,
        std::string *error) noexcept
    {
        if (error)
            error->clear();
        if (!batch.valid() ||
            batch.header.topology_fingerprint !=
                binding_.topology_fingerprint ||
            loadAcquire(binding_.transport->status_code) !=
                static_cast<std::uint32_t>(
                    MoEOverlayDeviceControllerError::None) ||
            loadAcquire(binding_.transport->command_transaction) !=
                batch.header.transaction_id ||
            binding_.transport->command_digest != batch.header.command_digest)
        {
            fail();
            return reject(
                error,
                "device transport completion does not match its acquired command");
        }
        return true;
    }

    bool MoEOverlayDeviceTransportProtocol::publishPrepared(
        const MoEOverlayDeviceTransportCommandBatch &batch,
        std::string *error) noexcept
    {
        if (!requireAcquired(batch, error))
            return false;
        binding_.transport->state = static_cast<std::uint32_t>(
            MoEOverlayDeviceControllerTransportState::Prepared);
        std::atomic_thread_fence(std::memory_order_release);
        storeRelease(
            binding_.transport->prepared_transaction,
            batch.header.transaction_id);
        return true;
    }

    bool MoEOverlayDeviceTransportProtocol::preparationReady(
        const MoEOverlayDeviceTransportCommandBatch &batch) const noexcept
    {
        if (!exactControllerTransaction(
                binding_, batch,
                MoEOverlayDeviceControllerState::PreparingFollowers))
        {
            return false;
        }
        if (!batch.movesWeights())
            return true;
        return loadAcquire(binding_.transport->command_transaction) ==
                   batch.header.transaction_id &&
               binding_.transport->command_digest ==
                   batch.header.command_digest &&
               loadAcquire(binding_.transport->prepared_transaction) ==
                   batch.header.transaction_id &&
               allParticipantsReached(
                   binding_, ParticipantPhase::Prepared,
                   batch.header.transaction_id);
    }

    bool MoEOverlayDeviceTransportProtocol::allGroupsPrepared(
        const MoEOverlayDeviceTransportCommandBatch &batch) const noexcept
    {
        return binding_.valid() && batch.valid() &&
               batch.header.topology_fingerprint ==
                   binding_.topology_fingerprint &&
               loadAcquire(binding_.controller->transaction_id) >=
                   batch.header.transaction_id &&
               allGroupsReached(
                   binding_, GroupPhase::Prepared,
                   batch.header.transaction_id);
    }

    bool MoEOverlayDeviceTransportProtocol::commitRequested(
        const MoEOverlayDeviceTransportCommandBatch &batch) const noexcept
    {
        return batch.valid() &&
               batch.header.topology_fingerprint ==
                   binding_.topology_fingerprint &&
               loadAcquire(binding_.controller->commit_transaction) ==
                   batch.header.transaction_id &&
               loadAcquire(binding_.controller->state) ==
                   static_cast<std::uint32_t>(
                       MoEOverlayDeviceControllerState::PublishingFollowers);
    }

    bool MoEOverlayDeviceTransportProtocol::publicationReady(
        const MoEOverlayDeviceTransportCommandBatch &batch) const noexcept
    {
        return commitRequested(batch) &&
               allParticipantsReached(
                   binding_,
                   ParticipantPhase::Published,
                   batch.header.transaction_id);
    }

    bool MoEOverlayDeviceTransportProtocol::authorityRejected(
        const MoEOverlayDeviceTransportCommandBatch &batch) const noexcept
    {
        return batch.valid() &&
               batch.header.topology_fingerprint ==
                   binding_.topology_fingerprint &&
               loadAcquire(binding_.controller->transaction_id) ==
                   batch.header.transaction_id &&
               (loadAcquire(binding_.controller->state) ==
                    static_cast<std::uint32_t>(
                        MoEOverlayDeviceControllerState::Error) ||
                localProtocolFailurePublished(binding_) ||
                topologyGroupFailurePublished(binding_));
    }

    bool MoEOverlayDeviceTransportProtocol::publishPublished(
        const MoEOverlayDeviceTransportCommandBatch &batch,
        std::string *error) noexcept
    {
        if (!requireAcquired(batch, error) ||
            loadAcquire(binding_.transport->prepared_transaction) !=
                batch.header.transaction_id ||
            (!exactControllerTransaction(
                 binding_,
                 batch,
                 MoEOverlayDeviceControllerState::PreparingFollowers) &&
             !exactControllerTransaction(
                 binding_,
                 batch,
                 MoEOverlayDeviceControllerState::PublishingFollowers)))
        {
            if (error && error->empty())
                *error = "device transport publication preceded complete physical preparation or fell outside the exact pre-admission publication epoch";
            if (loadAcquire(binding_.transport->status_code) == 0u)
                fail();
            return false;
        }
        binding_.transport->state = static_cast<std::uint32_t>(
            MoEOverlayDeviceControllerTransportState::Published);
        std::atomic_thread_fence(std::memory_order_release);
        storeRelease(
            binding_.transport->published_transaction,
            batch.header.transaction_id);
        return true;
    }

    bool MoEOverlayDeviceTransportProtocol::groupPublicationReady(
        const MoEOverlayDeviceTransportCommandBatch &batch) const noexcept
    {
        if (!commitRequested(batch))
            return false;
        if (!batch.movesWeights())
            return true;
        return loadAcquire(binding_.transport->published_transaction) ==
                   batch.header.transaction_id &&
               allParticipantsReached(
                   binding_, ParticipantPhase::Published,
                   batch.header.transaction_id);
    }

    bool MoEOverlayDeviceTransportProtocol::allGroupsPublished(
        const MoEOverlayDeviceTransportCommandBatch &batch) const noexcept
    {
        return binding_.valid() && batch.valid() &&
               batch.header.topology_fingerprint ==
                   binding_.topology_fingerprint &&
               loadAcquire(binding_.controller->transaction_id) >=
                   batch.header.transaction_id &&
               allGroupsReached(
                   binding_, GroupPhase::Published,
                   batch.header.transaction_id);
    }

    bool MoEOverlayDeviceTransportProtocol::retirementOpen(
        const MoEOverlayDeviceTransportCommandBatch &batch) const noexcept
    {
        return exactControllerTransaction(
                   binding_, batch,
                   MoEOverlayDeviceControllerState::RetiringDurableEpoch) &&
               loadAcquire(binding_.controller->admission_transaction) ==
                   batch.header.transaction_id &&
               loadAcquire(binding_.controller->admission_epoch) ==
                   batch.header.candidate_epoch;
    }

    bool MoEOverlayDeviceTransportProtocol::retirementRequested(
        const MoEOverlayDeviceTransportCommandBatch &batch) const noexcept
    {
        return retirementOpen(batch) &&
               durablePlacementKind(static_cast<
                   MoEOverlayDeviceControllerTransactionKind>(
                   batch.header.kind)) &&
               (!batch.movesWeights() ||
                allParticipantsReached(
                    binding_,
                    ParticipantPhase::Retired,
                    batch.header.base_epoch));
    }

    bool MoEOverlayDeviceTransportProtocol::runtimeReadersReady(
        const MoEOverlayDeviceTransportCommandBatch &batch) const noexcept
    {
        return retirementOpen(batch) && batch.movesWeights() &&
               batch.header.candidate_epoch > batch.header.base_epoch &&
               allTopologyParticipantsReached(
                   binding_,
                   ParticipantPhase::RetirementReady,
                   batch.header.base_epoch);
    }

    bool MoEOverlayDeviceTransportProtocol::publishRetired(
        const MoEOverlayDeviceTransportCommandBatch &batch,
        std::string *error) noexcept
    {
        if (!requireAcquired(batch, error) ||
            !durablePlacementKind(static_cast<
                MoEOverlayDeviceControllerTransactionKind>(
                batch.header.kind)) ||
            batch.header.candidate_epoch == batch.header.base_epoch ||
            loadAcquire(binding_.transport->published_transaction) !=
                batch.header.transaction_id ||
            loadAcquire(binding_.controller->state) !=
                static_cast<std::uint32_t>(
                    MoEOverlayDeviceControllerState::RetiringDurableEpoch) ||
            !allParticipantsReached(
                binding_,
                ParticipantPhase::Retired,
                batch.header.base_epoch))
        {
            if (error && error->empty())
                *error = "device transport retirement preceded publication or retirement admission";
            if (loadAcquire(binding_.transport->status_code) == 0u)
                fail();
            return false;
        }
        binding_.transport->state = static_cast<std::uint32_t>(
            MoEOverlayDeviceControllerTransportState::Retired);
        std::atomic_thread_fence(std::memory_order_release);
        storeRelease(
            binding_.transport->retired_epoch,
            batch.header.base_epoch);
        return true;
    }

    bool MoEOverlayDeviceTransportProtocol::groupRetirementReady(
        const MoEOverlayDeviceTransportCommandBatch &batch) const noexcept
    {
        if (!retirementRequested(batch))
            return false;
        if (!batch.movesWeights())
            return true;
        return loadAcquire(binding_.transport->retired_epoch) ==
            batch.header.base_epoch;
    }

    bool MoEOverlayDeviceTransportProtocol::allGroupsRetired(
        const MoEOverlayDeviceTransportCommandBatch &batch) const noexcept
    {
        return binding_.valid() && batch.valid() &&
               batch.header.topology_fingerprint ==
                   binding_.topology_fingerprint &&
               durablePlacementKind(static_cast<
                   MoEOverlayDeviceControllerTransactionKind>(
                   batch.header.kind)) &&
               loadAcquire(binding_.controller->transaction_id) >=
                   batch.header.transaction_id &&
               allGroupsReached(
                   binding_, GroupPhase::Retired,
                   batch.header.base_epoch);
    }

    bool MoEOverlayDeviceTransportProtocol::transactionComplete(
        const MoEOverlayDeviceTransportCommandBatch &batch) const noexcept
    {
        return binding_.valid() && batch.valid() &&
               batch.header.topology_fingerprint ==
                   binding_.topology_fingerprint &&
               loadAcquire(binding_.controller->completed_transaction) >=
                   batch.header.transaction_id;
    }

    bool MoEOverlayDeviceTransportProtocol::restorationRequested(
        const MoEOverlayDeviceTransportCommandBatch &batch) const noexcept
    {
        return batch.valid() &&
               batch.header.kind == static_cast<std::uint32_t>(
                   MoEOverlayDeviceControllerTransactionKind::
                       CurrentBatchLLEP) &&
               loadAcquire(binding_.controller->transaction_id) ==
                   batch.header.transaction_id &&
               loadAcquire(binding_.controller->state) ==
                   static_cast<std::uint32_t>(
                       MoEOverlayDeviceControllerState::RestoringLLEP) &&
               allParticipantsReached(
                   binding_,
                   ParticipantPhase::Restored,
                   batch.header.transaction_id);
    }

    std::string MoEOverlayDeviceTransportProtocol::describeLifecycle() const
    {
        std::ostringstream description;
        description << "group=" << binding_.group_id;
        if (!binding_.valid())
        {
            description << ",binding=invalid";
            return description.str();
        }

        const auto *controller = binding_.controller;
        description
            << ",controller{state=" << loadAcquire(controller->state)
            << ",transaction=" << loadAcquire(controller->transaction_id)
            << ",kind=" << loadAcquire(controller->transaction_kind)
            << ",intent_phase="
            << loadAcquire(controller->transaction_demand_phase)
            << ",durable="
            << loadAcquire(controller->current_durable_epoch)
            << ",base=" << loadAcquire(controller->base_epoch)
            << ",candidate=" << loadAcquire(controller->candidate_epoch)
            << ",command="
            << loadAcquire(controller->command_transaction)
            << ",commit=" << loadAcquire(controller->commit_transaction)
            << ",admission="
            << loadAcquire(controller->admission_transaction)
            << ",admission_epoch="
            << loadAcquire(controller->admission_epoch)
            << ",completed="
            << loadAcquire(controller->completed_transaction)
            << ",placement_layer_cursor="
            << loadAcquire(controller->placement_layer_cursor)
            << ",error=" << loadAcquire(controller->error_code)
            << ",error_group=" << controller->error_group_id << '}';

        const auto *command = binding_.command;
        description
            << ",command_header{transaction="
            << loadAcquire(command->transaction_id)
            << ",count=" << loadAcquire(command->command_count)
            << ",demand_phase=" << loadAcquire(command->demand_phase)
            << ",scan_start=" << loadAcquire(command->layer_scan_start)
            << ",scan_next=" << loadAcquire(command->layer_scan_next)
            << '}';

        const auto *transport = binding_.transport;
        description
            << ",transport{state=" << loadAcquire(transport->state)
            << ",status=" << loadAcquire(transport->status_code)
            << ",command="
            << loadAcquire(transport->command_transaction)
            << ",prepared="
            << loadAcquire(transport->prepared_transaction)
            << ",published="
            << loadAcquire(transport->published_transaction)
            << ",retired=" << loadAcquire(transport->retired_epoch)
            << '}';

        description << ",groups=[";
        for (std::uint32_t index = 0u;
             index < binding_.group_record_count;
             ++index)
        {
            if (index != 0u)
                description << ';';
            const auto *record = binding_.group_records[index];
            if (!record)
            {
                description << index << ":null";
                continue;
            }
            description
                << index << ":status=" << loadAcquire(record->status_code)
                << ",snapshot="
                << loadAcquire(record->snapshot_transaction)
                << ",prepared="
                << loadAcquire(record->prepared_transaction)
                << ",published="
                << loadAcquire(record->published_transaction)
                << ",retired=" << loadAcquire(record->retired_epoch);
        }
        description << "],participants=[";
        // A peer can time out after every local participant has progressed.
        // Inspect every already-mapped lifecycle lane so that its diagnostic
        // identifies the missing remote action/receipt too. These observations
        // never authorize a transition and expose no runtime placement state.
        for (std::uint32_t group = 0u;
             group < binding_.group_record_count;
             ++group)
        {
            if (group != 0u)
                description << ';';
            description << "group=" << group << '{';
            // valid() above proves both the array bounds and each lane's
            // presence before any remote pointer can be dereferenced.
            for (std::uint32_t index = 0u;
                 index < binding_.topology_participant_record_counts[group];
                 ++index)
            {
                if (index != 0u)
                    description << ';';
                const auto &record =
                    binding_.topology_participant_records[group][index];
                description
                    << record.participant_id
                    << ":status=" << loadAcquire(record.status_code)
                    << ",entered_action=" << loadAcquire(record.observed_action)
                    << ",entry_transaction="
                    << loadAcquire(record.observed_action_transaction)
                    << ",snapshot="
                    << loadAcquire(record.snapshot_transaction)
                    << ",prepared="
                    << loadAcquire(record.prepared_transaction)
                    << ",published="
                    << loadAcquire(record.published_transaction)
                    << ",ready="
                    << loadAcquire(record.retirement_ready_epoch)
                    << ",retired=" << loadAcquire(record.retired_epoch);
            }
            description << '}';
        }
        description << ']';
        return description.str();
    }

    bool MoEOverlayDeviceTransportProtocol::publishRestored(
        const MoEOverlayDeviceTransportCommandBatch &batch,
        std::string *error) noexcept
    {
        if (!requireAcquired(batch, error) ||
            batch.header.kind != static_cast<std::uint32_t>(
                MoEOverlayDeviceControllerTransactionKind::CurrentBatchLLEP) ||
            loadAcquire(binding_.transport->published_transaction) !=
                batch.header.transaction_id ||
            loadAcquire(binding_.controller->state) !=
                static_cast<std::uint32_t>(
                    MoEOverlayDeviceControllerState::RestoringLLEP) ||
            !allParticipantsReached(
                binding_,
                ParticipantPhase::Restored,
                batch.header.transaction_id))
        {
            if (error && error->empty())
                *error = "device transport restoration preceded publication or restore admission";
            if (loadAcquire(binding_.transport->status_code) == 0u)
                fail();
            return false;
        }
        binding_.transport->state = static_cast<std::uint32_t>(
            MoEOverlayDeviceControllerTransportState::Restored);
        std::atomic_thread_fence(std::memory_order_release);
        storeRelease(
            binding_.transport->restored_transaction,
            batch.header.transaction_id);
        return true;
    }

    void MoEOverlayDeviceTransportProtocol::fail() noexcept
    {
        if (!binding_.transport)
            return;
        binding_.transport->state = static_cast<std::uint32_t>(
            MoEOverlayDeviceControllerTransportState::Error);
        std::atomic_thread_fence(std::memory_order_release);
        storeRelease(
            binding_.transport->status_code,
            static_cast<std::uint32_t>(
                MoEOverlayDeviceControllerError::PhysicalTransportFailure));
    }
} // namespace llaminar2
