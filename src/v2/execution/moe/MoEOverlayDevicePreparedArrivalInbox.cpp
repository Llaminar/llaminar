/**
 * @file MoEOverlayDevicePreparedArrivalInbox.cpp
 * @brief Immutable mapped descriptor handoff for device-authored MoE movement.
 *
 * The physical transport worker writes descriptors once, then releases its
 * existing transaction receipt. The captured controller acquires that receipt
 * and installs descriptors into an inactive device bank. A second H2D copy and
 * completion event are unnecessary and can be serialized behind live inference
 * by native GPU work queues. Pages remain immutable until device completion.
 */

#include "MoEOverlayDevicePreparedArrivalInbox.h"

#include "DeviceMoEExpertDescriptorBuilder.h"
#include "MoEExpertShardCohort.h"
#include "MoEIntermediateShardDescriptorDeclaration.h"
#include "MoERuntimeTable.h"
#include "backends/IBackend.h"
#include "MoEOverlayDeviceTransportProtocol.h"
#include "transfer/TransferEngine.h"
#include "utils/PerfStatsCollector.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <stdexcept>
#include <utility>

namespace llaminar2
{
    namespace
    {
        /** Populate an optional diagnostic and return false for guard clauses. */
        bool reject(std::string *error, std::string message) noexcept
        {
            if (error)
                *error = std::move(message);
            return false;
        }

        /**
         * @brief Export one completed durable physical-ledger allocation.
         *
         * ExpertOverlay shadow slots are retained by the model-lifetime
         * physical ledger. They are not entries in
         * `DeviceMoETransferSlotDirectory`, whose `TransferSlot` identity is
         * reserved for request-scoped LLEP/prefix-rehydration payloads. Use the
         * normal logical-expert slot convention here so reset, prefix capture,
         * and device storage-liveness code cannot mistake durable placement for
         * transient transfer storage.
         *
         * The controller overwrites owner and executable flags only after every
         * participant has authenticated this wave. Publishing Valid/Resident
         * here is sufficient for the prepared-arrival readiness predicate while
         * keeping the descriptor ineligible for local compute before RCU apply.
         */
        bool exportArrival(
            const MoEOverlayPreparedExpertPayload &payload,
            int expert,
            DeviceMoEExpertDescriptor *output) noexcept
        {
            if (!output || expert < 0 || !payload.ready())
            {
                return false;
            }
            DeviceMoEExpertDescriptor descriptor{};
            if (!exportDeviceMoEPreparedPayload(payload, descriptor))
            {
                return false;
            }
            descriptor.logical_expert_id = expert;
            descriptor.owner_participant = -1;
            descriptor.local_slot = expert;
            descriptor.flags = toMoEExpertFlags(
                DeviceMoEExpertFlags::Valid |
                DeviceMoEExpertFlags::Resident);
            *output = descriptor;
            return true;
        }
    } // namespace

    MoEOverlayDevicePreparedArrivalInbox::
        MoEOverlayDevicePreparedArrivalInbox(Config config)
        : config_(std::move(config))
    {
        if (!config_.backend || !config_.runtime_binding.publicationValid() ||
            config_.command_capacity == 0u || !config_.consumer_stream)
        {
            throw std::invalid_argument(
                "device prepared-arrival inbox requires a complete GPU publication binding, positive capacity, and exact consumer stream");
        }
        if (const auto *original = config_.runtime_binding.runtime_table_host;
            original && !config_.runtime_binding.matchesLayerScope(
                original->firstModelLayer(), static_cast<std::uint32_t>(original->layerCount())))
            throw std::invalid_argument("device prepared-arrival inbox changed its original runtime layer scope");

        const int ordinal = config_.runtime_binding.device.gpu_ordinal();
        const std::size_t arrival_bytes =
            static_cast<std::size_t>(config_.command_capacity) *
            sizeof(DeviceMoEExpertDescriptor);
        try
        {
            const std::array<DeviceId, 1u> devices{
                config_.runtime_binding.device};
            mapped_arrivals_ =
                TransferEngine::instance().allocateMappedHostRegion(
                    arrival_bytes, devices);
            prepared_arrivals_device_ = static_cast<DeviceMoEExpertDescriptor *>(
                mapped_arrivals_->deviceAlias(config_.runtime_binding.device));
            prepared_arrivals_staging_ = static_cast<DeviceMoEExpertDescriptor *>(
                mapped_arrivals_->mutableHostData());
            apply_status_device_ = static_cast<
                MoEOverlayDeviceRuntimeApplyStatus *>(
                config_.backend->allocate(
                    sizeof(MoEOverlayDeviceRuntimeApplyStatus), ordinal));
            if (!prepared_arrivals_device_ || !prepared_arrivals_staging_ ||
                !apply_status_device_)
            {
                throw std::runtime_error(
                    "device prepared-arrival inbox setup allocation failed");
            }

            std::fill_n(
                prepared_arrivals_staging_,
                config_.command_capacity,
                DeviceMoEExpertDescriptor{});
            // The status consumer uses this same setup stream. Host descriptor
            // pages require no GPU initialization or cross-stream event edge.
            if (!config_.backend->memset(
                    apply_status_device_,
                    0,
                    sizeof(MoEOverlayDeviceRuntimeApplyStatus),
                    ordinal,
                    config_.consumer_stream))
            {
                throw std::runtime_error(
                    "device prepared-arrival inbox setup initialization failed");
            }
        }
        catch (...)
        {
            release();
            throw;
        }
    }

    MoEOverlayDevicePreparedArrivalInbox::
        ~MoEOverlayDevicePreparedArrivalInbox()
    {
        release();
    }

    MoEOverlayDeviceRuntimePublicationBinding
    MoEOverlayDevicePreparedArrivalInbox::publicationBinding() const noexcept
    {
        return {
            .runtime_layers =
                config_.runtime_binding.runtime_layers_device,
            .epoch_control = config_.runtime_binding.epoch_control,
            .epoch_status = config_.runtime_binding.maintenance_status,
            .prepared_arrivals = prepared_arrivals_device_,
            .apply_status = apply_status_device_,
            .arrival_capacity = config_.command_capacity,
            .layer_count = config_.runtime_binding.layer_count,
            .expert_count = config_.runtime_binding.expert_count,
            .domain_participant_id =
                config_.runtime_binding.domain_participant_id,
            .domain_participant_count =
                config_.runtime_binding.domain_participant_count,
        };
    }

    bool MoEOverlayDevicePreparedArrivalInbox::exportDestinationArrival(
        const MoEOverlayTierMigration &migration,
        std::span<const MoEOverlayParticipantDestinationArrival> arrivals,
        std::uint64_t candidate_epoch, int participant_id,
        const MoEIntermediateShardDescriptorDeclaration *source,
        std::optional<DeviceMoEExpertDescriptor> &output,
        std::string *error) noexcept
    {
        if (error)
            error->clear();
        const auto target = MoEExpertPlacementTarget::fromPlacement(migration.destination);
        if (!target || candidate_epoch == 0u || participant_id < 0 ||
            migration.layer_idx < 0 || migration.expert_id < 0 ||
            migration.destination.layer_idx != migration.layer_idx ||
            migration.destination.expert_id != migration.expert_id)
            return reject(error, "device arrival has no exact original destination target");

        bool is_local = false;
        for (std::size_t member = 0; member < target->participantCount(); ++member)
            is_local = is_local || target->participantId(member) == participant_id;
        const MoEOverlayPreparedExpertArrival *local = nullptr;
        for (std::size_t index = 0; index < arrivals.size(); ++index)
        {
            const auto &entry = arrivals[index];
            bool belongs = false;
            for (std::size_t member = 0; member < target->participantCount(); ++member)
                belongs = belongs || target->participantId(member) == entry.participant_id;
            if (!belongs || !entry.arrival ||
                std::any_of(arrivals.begin(), arrivals.begin() + index,
                    [&entry](const auto &previous) { return previous.participant_id == entry.participant_id; }))
                return reject(error, "device arrival repeats or replaces an original destination member");
            if (entry.arrival->isJointFragment() != target->isCohort())
                return reject(error, "device arrival changed the original whole/cohort payload arm");
            if (entry.participant_id == participant_id)
                local = entry.arrival.get();
        }
        if (!is_local)
        {
            // The rank can retain incomplete sibling arrivals. This inbox owns
            // no payload inspection or descriptor publication for those peers.
            output.reset();
            return true;
        }
        if (!local)
            return reject(error, "device arrival is missing the exact local destination member");

        DeviceMoEExpertDescriptor descriptor{};
        if (target->isCohort())
        {
            MoEOverlayPreparedFragmentPayload fragment;
            const auto *family = source ? source->originalFamily() : nullptr;
            if (!family)
                return reject(error, "device fragment arrival has no original runtime-bound family");
            if (local->destinationEpoch() != candidate_epoch)
                return reject(error, "device fragment arrival belongs to a different candidate epoch");
            if (!local->completeFragment(fragment, error))
                return false;
            const auto &cohort = migration.destination.shard_cohort;
            if (!family->accepts(fragment, migration.expert_id) ||
                fragment.arrivalEpoch() != candidate_epoch)
                return reject(error, "device fragment arrival replaced its original family, expert or arrival epoch");
            if (fragment.cohort() != cohort || fragment.cohort().owner_before(cohort) ||
                cohort.owner_before(fragment.cohort()) || cohort->source().layer_idx != migration.layer_idx)
                return reject(error, "device fragment arrival replaced its original source cohort or model layer");
            if (fragment.participantIndex() >= cohort->participants().size() ||
                cohort->participants()[fragment.participantIndex()].participant_id != participant_id ||
                !cohort->participants()[fragment.participantIndex()].address.isGPU())
                return reject(error, "device fragment arrival changed its original global GPU member");
            if (!exportDeviceMoEFragmentWeightDescriptors(fragment, descriptor))
                return reject(error, "device fragment arrival has incomplete native projection geometry");
            if (!matchesExportedFragment(*source, descriptor))
                return reject(error, "device fragment arrival disagrees with its original source precision, physical planes or local geometry");
            descriptor.logical_expert_id = migration.expert_id;
            descriptor.owner_participant = -1;
            descriptor.local_slot = migration.expert_id;
            // Prepared bytes are not executable until candidate publication.
            descriptor.flags = toMoEExpertFlags(DeviceMoEExpertFlags::Valid | DeviceMoEExpertFlags::Resident);
        }
        else if (source)
            return reject(error, "device whole arrival cannot occupy an original fragment lane");
        else
        {
            MoEOverlayPreparedExpertPayload payload;
            if (!local->completePayload(payload, error))
                return false;
            if (!exportArrival(payload, migration.expert_id, &descriptor))
                return reject(error, "device whole arrival has no complete native destination payload");
        }
        output = descriptor;
        return true;
    }

    bool MoEOverlayDevicePreparedArrivalInbox::matchesExportedFragment(
        const MoEIntermediateShardDescriptorDeclaration &source,
        const DeviceMoEExpertDescriptor &descriptor) noexcept
    {
        if (!source.valid() || descriptor.projection_set != DeviceMoEProjectionSet::IntermediateShard ||
            descriptor.weight_format != source.weightFormat())
            return false;
        if (descriptor.weight_format == DeviceMoEWeightFormat::NativeQuantised)
            return source.matchesPublishedNativeGateUp(descriptor.gate, descriptor.up) &&
                source.matchesPublishedNativeDown(descriptor.down);
        return source.matchesFloatingGateUp(descriptor.floating_gate, descriptor.floating_up) &&
            source.matchesFloatingDown(descriptor.floating_down);
    }

    bool MoEOverlayDevicePreparedArrivalInbox::stage(
        const MoEOverlayDevicePhysicalMovementBatch &batch,
        const MoEOverlayParticipantPreparedTransfers &prepared,
        std::string *error) noexcept
    {
        if (error)
            error->clear();
        if (state_ != State::Ready ||
            !batch.valid() ||
            !config_.runtime_binding.matchesLayerScope(
                batch.first_model_layer, batch.num_layers) ||
            config_.runtime_binding.expert_count != batch.num_experts ||
            batch.transaction_id <= retired_transaction_ ||
            !batch.movesWeights() ||
            (batch.kind !=
                 MoEOverlayDeviceControllerTransactionKind::DynamicPlacement &&
             batch.kind != MoEOverlayDeviceControllerTransactionKind::
                               PreparedContextRestore) ||
            batch.command_count > config_.command_capacity ||
            batch.migrations.size() != batch.command_count ||
            prepared.status != MoEOverlayResidencyStageStartStatus::Started ||
            prepared.migrations.size() != batch.migrations.size())
        {
            return reject(
                error,
                "device prepared-arrival inbox rejected invalid or overlapping physical movement");
        }

        std::fill_n(prepared_arrivals_staging_, batch.command_count, DeviceMoEExpertDescriptor{});
        std::uint32_t local_arrivals = 0u;
        // Ready owns no captured reader of these pages. Failed staging never
        // releases a transport receipt, so a rejected prefix is unpublished.
        for (std::size_t ordinal = 0u; ordinal < batch.migrations.size(); ++ordinal)
        {
            const auto &migration = batch.migrations[ordinal];
            const auto &keyed = prepared.migrations[ordinal].destination_arrivals;
            const auto *source = config_.runtime_binding.runtime_table_host
                ? config_.runtime_binding.runtime_table_host->fragmentDescriptorDeclaration(migration.layer_idx) : nullptr;
            std::optional<DeviceMoEExpertDescriptor> descriptor;
            if (!exportDestinationArrival(migration, keyed, batch.candidate_epoch,
                    config_.runtime_binding.overlay_participant_id, source, descriptor, error))
                return false;
            prepared_arrivals_staging_[ordinal] = descriptor.value_or(DeviceMoEExpertDescriptor{});
            local_arrivals += descriptor.has_value() ? 1u : 0u;
        }

        const std::size_t bytes =
            static_cast<std::size_t>(batch.command_count) *
            sizeof(DeviceMoEExpertDescriptor);
        // publishPrepared() follows on this transport worker. Its system-release
        // receipt is the only publication edge; device candidate construction
        // acquires it before reading these immutable physical descriptors.
        staged_transaction_ = batch.transaction_id;
        staged_digest_ = batch.command_digest;
        state_ = State::Staged;
        PerfStatsCollector::addCounter(
            "moe_overlay_controller",
            "prepared_arrival_descriptor_publications",
            1.0,
            "maintenance",
            config_.perf_device,
            {{"participant",
              std::to_string(
                  config_.runtime_binding.overlay_participant_id)},
             {"commands", std::to_string(batch.command_count)},
             {"local_arrivals", std::to_string(local_arrivals)},
             {"bytes", std::to_string(bytes)},
             {"handoff", "mapped_immutable"},
             {"gpu_submissions", "0"},
             {"blocking", "false"}});
        return true;
    }

    bool MoEOverlayDevicePreparedArrivalInbox::finishWave(
        const MoEOverlayDeviceTransportProtocol &protocol,
        const MoEOverlayDeviceTransportCommandBatch &command,
        std::string *error) noexcept
    {
        if (state_ != State::Staged ||
            command.header.transaction_id != staged_transaction_ ||
            command.header.command_digest != staged_digest_ ||
            !protocol.transactionComplete(command))
        {
            return reject(error,
                "device prepared-arrival inbox requires its exact completed controller transaction before reuse");
        }
        retired_transaction_ = staged_transaction_;
        staged_transaction_ = 0u;
        staged_digest_ = 0u;
        state_ = State::Ready;
        return true;
    }

    void MoEOverlayDevicePreparedArrivalInbox::release() noexcept
    {
        if (!config_.backend)
            return;
        const int ordinal = config_.runtime_binding.device.gpu_ordinal();
        // The service retires captured readers before releasing its endpoints.
        // This inbox has no independent queued work or completion event to drain.
        if (apply_status_device_)
            config_.backend->free(apply_status_device_, ordinal);
        mapped_arrivals_.reset();
        apply_status_device_ = nullptr;
        prepared_arrivals_staging_ = nullptr;
        prepared_arrivals_device_ = nullptr;
        state_ = State::Released;
    }
} // namespace llaminar2
