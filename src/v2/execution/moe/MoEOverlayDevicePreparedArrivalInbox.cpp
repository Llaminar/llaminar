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
            const MoEOverlayPreparedExpertTriplet &triplet,
            int expert,
            DeviceMoEExpertDescriptor *output) noexcept
        {
            if (!output || expert < 0 || !triplet.complete())
            {
                return false;
            }
            DeviceMoEExpertDescriptor descriptor{};
            if (!exportDeviceMoEExpertWeightDescriptors(
                    triplet.gate.get(),
                    triplet.up.get(),
                    triplet.down.get(),
                    descriptor))
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

    bool MoEOverlayDevicePreparedArrivalInbox::stage(
        const MoEOverlayDevicePhysicalMovementBatch &batch,
        const MoEOverlayParticipantPreparedTransfers &prepared,
        std::string *error) noexcept
    {
        if (error)
            error->clear();
        if (state_ != State::Ready ||
            !batch.valid() ||
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

        std::fill_n(
            prepared_arrivals_staging_,
            batch.command_count,
            DeviceMoEExpertDescriptor{});
        std::uint32_t local_arrivals = 0u;
        for (std::size_t ordinal = 0u;
             ordinal < batch.migrations.size();
             ++ordinal)
        {
            const auto &migration = batch.migrations[ordinal];
            const auto &arrival =
                prepared.migrations[ordinal].destination_arrival;
            const bool local_destination =
                migration.destination.owner_participant ==
                config_.runtime_binding.overlay_participant_id;
            if (!local_destination)
            {
                /*
                 * `prepared` is process-local, not participant-filtered. A
                 * rank with several GPU participants therefore retains the
                 * sibling destination's lifetime in this same vector. Only
                 * the matching inbox exports it; every other inbox leaves the
                 * command ordinal zero so its apply graph cannot consume a
                 * foreign descriptor. Rank ownership was already validated by
                 * stageDevicePreparedTransfers before any inbox is enqueued.
                 */
                continue;
            }
            if (!arrival)
            {
                return reject(
                    error,
                    "device prepared-arrival inbox lost a local destination lifetime");
            }
            MoEOverlayPreparedExpertTriplet triplet;
            std::string arrival_error;
            if (!arrival->completeTriplet(triplet, &arrival_error) ||
                !exportArrival(
                    triplet,
                    migration.expert_id,
                    prepared_arrivals_staging_ + ordinal))
            {
                return reject(
                    error,
                    arrival_error.empty()
                        ? "device prepared-arrival inbox could not export a complete local descriptor"
                        : std::move(arrival_error));
            }
            ++local_arrivals;
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
