/**
 * @file MoEOverlayDevicePreparedArrivalInbox.cpp
 * @brief Exact-stream descriptor handoff for device-authored MoE movement.
 */

#include "MoEOverlayDevicePreparedArrivalInbox.h"

#include "DeviceMoEExpertDescriptorBuilder.h"
#include "backends/IBackend.h"
#include "utils/PerfStatsCollector.h"

#include <algorithm>
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
            prepared_arrivals_device_ = static_cast<DeviceMoEExpertDescriptor *>(
                config_.backend->allocate(arrival_bytes, ordinal));
            prepared_arrivals_staging_ = static_cast<DeviceMoEExpertDescriptor *>(
                config_.backend->allocatePinned(arrival_bytes, ordinal));
            apply_status_device_ = static_cast<
                MoEOverlayDeviceRuntimeApplyStatus *>(
                config_.backend->allocate(
                    sizeof(MoEOverlayDeviceRuntimeApplyStatus), ordinal));
            transfer_stream_ = config_.backend->createStream(ordinal);
            transfer_event_ = config_.backend->createEvent(ordinal);
            if (!prepared_arrivals_device_ || !prepared_arrivals_staging_ ||
                !apply_status_device_ || !transfer_stream_ || !transfer_event_)
            {
                throw std::runtime_error(
                    "device prepared-arrival inbox setup allocation failed");
            }

            std::fill_n(
                prepared_arrivals_staging_,
                config_.command_capacity,
                DeviceMoEExpertDescriptor{});
            if (!config_.backend->memset(
                    prepared_arrivals_device_,
                    0,
                    arrival_bytes,
                    ordinal,
                    transfer_stream_) ||
                !config_.backend->memset(
                    apply_status_device_,
                    0,
                    sizeof(MoEOverlayDeviceRuntimeApplyStatus),
                    ordinal,
                    transfer_stream_) ||
                !config_.backend->recordEvent(
                    transfer_event_, ordinal, transfer_stream_) ||
                !config_.backend->streamWaitEvent(
                    config_.consumer_stream,
                    transfer_event_,
                    ordinal))
            {
                throw std::runtime_error(
                    "device prepared-arrival inbox setup initialization failed");
            }
            state_ = State::InitializationSubmitted;
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

    bool MoEOverlayDevicePreparedArrivalInbox::enqueue(
        const MoEOverlayDevicePhysicalMovementBatch &batch,
        const MoEOverlayParticipantPreparedTransfers &prepared,
        std::string *error) noexcept
    {
        if (error)
            error->clear();
        if ((state_ != State::InitializationSubmitted &&
             state_ != State::Ready) ||
            !batch.valid() ||
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

        const int ordinal = config_.runtime_binding.device.gpu_ordinal();
        const std::size_t bytes =
            static_cast<std::size_t>(batch.command_count) *
            sizeof(DeviceMoEExpertDescriptor);
        if (!config_.backend->hostToDeviceOnStream(
                prepared_arrivals_device_,
                prepared_arrivals_staging_,
                bytes,
                ordinal,
                transfer_stream_) ||
            !config_.backend->recordEvent(
                transfer_event_, ordinal, transfer_stream_))
        {
            return reject(
                error,
                "device prepared-arrival inbox descriptor DMA submission failed");
        }
        state_ = State::WaveSubmitted;
        PerfStatsCollector::addCounter(
            "moe_overlay_controller",
            "prepared_arrival_descriptor_uploads",
            1.0,
            "maintenance",
            config_.perf_device,
            {{"participant",
              std::to_string(
                  config_.runtime_binding.overlay_participant_id)},
             {"commands", std::to_string(batch.command_count)},
             {"local_arrivals", std::to_string(local_arrivals)},
             {"bytes", std::to_string(bytes)},
             {"blocking", "false"}});
        return true;
    }

    bool MoEOverlayDevicePreparedArrivalInbox::enqueueDependency(
        std::string *error) const noexcept
    {
        if (state_ != State::WaveSubmitted || !transfer_event_ ||
            !config_.backend->streamWaitEvent(
                config_.consumer_stream,
                transfer_event_,
                config_.runtime_binding.device.gpu_ordinal()))
        {
            return reject(
                error,
                "device controller stream could not join its prepared-arrival event");
        }
        return true;
    }

    bool MoEOverlayDevicePreparedArrivalInbox::queryReady(
        bool *ready,
        std::string *error) const noexcept
    {
        if (ready)
            *ready = false;
        if (!ready || state_ != State::WaveSubmitted || !transfer_event_ ||
            !config_.backend->queryEvent(
                transfer_event_,
                config_.runtime_binding.device.gpu_ordinal(),
                ready))
        {
            return reject(
                error,
                "device prepared-arrival inbox event query failed");
        }
        return true;
    }

    bool MoEOverlayDevicePreparedArrivalInbox::finishWave(
        std::string *error) noexcept
    {
        bool ready = false;
        if (!queryReady(&ready, error) || !ready)
        {
            if (error && error->empty())
                *error = "device prepared-arrival inbox was reused before DMA completion";
            return false;
        }
        state_ = State::Ready;
        return true;
    }

    void MoEOverlayDevicePreparedArrivalInbox::release() noexcept
    {
        if (!config_.backend)
            return;
        const int ordinal = config_.runtime_binding.device.gpu_ordinal();
        /* Initialization and a live wave can still own device/pinned storage.
         * Teardown is the only host wait: normal reuse observes completion by
         * query and reaches Ready before any resource is reclaimed. */
        if ((state_ == State::InitializationSubmitted ||
             state_ == State::WaveSubmitted) &&
            transfer_event_)
            (void)config_.backend->waitForEvent(transfer_event_, ordinal);
        if (transfer_event_)
            config_.backend->destroyEvent(transfer_event_, ordinal);
        if (transfer_stream_)
            config_.backend->destroyStream(transfer_stream_, ordinal);
        if (apply_status_device_)
            config_.backend->free(apply_status_device_, ordinal);
        if (prepared_arrivals_staging_)
            config_.backend->freePinned(
                prepared_arrivals_staging_, ordinal);
        if (prepared_arrivals_device_)
            config_.backend->free(prepared_arrivals_device_, ordinal);
        transfer_event_ = nullptr;
        transfer_stream_ = nullptr;
        apply_status_device_ = nullptr;
        prepared_arrivals_staging_ = nullptr;
        prepared_arrivals_device_ = nullptr;
        state_ = State::Released;
    }
} // namespace llaminar2
