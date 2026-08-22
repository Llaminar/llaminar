/**
 * @file MoEOverlayDeviceRuntimePublicationFixture.h
 * @brief Real-device fixture for topology-controller runtime publication.
 *
 * The fixture presents the same model-owned runtime table, device RCU arena,
 * transfer-prepared descriptor inbox, and exact transfer stream used by the
 * production publication protocol. Setup may wait for initialization; epoch
 * movement uses a persistent pinned host buffer, asynchronous DMA, and one
 * polled event so it never synchronizes an inference or maintenance stream.
 */

#pragma once

#include "backends/IBackend.h"
#include "execution/moe/DeviceMoEOverlayEpochArena.h"
#include "execution/moe/DeviceMoERebalancePolicyShared.h"
#include "execution/moe/MoEOverlayDeviceControllerKernels.h"
#include "execution/moe/MoEOverlayDeviceControllerRuntimeBinding.h"
#include "execution/moe/MoEOverlayDeviceControllerTopology.h"
#include "execution/moe/MoEOverlayDevicePlacementPolicy.h"
#include "execution/moe/MoEOverlayDeviceTransportProtocol.h"
#include "execution/moe/MoERuntimeTable.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace llaminar2::test
{
    /**
     * @brief Own one participant's production-shaped inactive-bank resources.
     *
     * The initial runtime is derived from the exact topology-wide packed
     * snapshot. Every group member therefore agrees on global routing while
     * only the authoritative local participant owns executable weight pointers.
     */
    class MoEOverlayDeviceRuntimePublicationFixture final
    {
    public:
        /** Construct and publish one exact participant view of the base epoch. */
        MoEOverlayDeviceRuntimePublicationFixture(
            IBackend *backend,
            const MoEExpertOwnerParticipant &participant,
            const MoEOverlayDeviceControllerTopology &topology,
            const MoEOverlayDevicePlacementPolicyInput &input,
            const std::uint64_t *external_admission_epoch = nullptr,
            std::shared_ptr<const void> external_admission_lifetime = {})
            : backend_(backend),
              participant_(participant),
              topology_(&topology),
              arrival_capacity_(input.command_capacity)
        {
            const auto *const group = topology.groupForParticipant(
                participant.participant_id);
            if (!backend_ || !participant.device.is_gpu() || !group ||
                input.num_layers == 0u || input.num_experts == 0u ||
                input.command_capacity == 0u || input.base_epoch == 0u ||
                participant.domain_participant_index < 0)
            {
                throw std::invalid_argument(
                    "publication runtime fixture requires complete participant topology and policy geometry");
            }

            const int ordinal = participant.device.ordinal;
            const std::size_t runtime_bytes =
                static_cast<std::size_t>(input.num_layers) *
                sizeof(DeviceMoELayerRuntime);
            const std::size_t arrival_bytes =
                static_cast<std::size_t>(arrival_capacity_) *
                sizeof(DeviceMoEExpertDescriptor);
            try
            {
                dummy_payload_ = backend_->allocate(256u, ordinal);
                dummy_scales_ = backend_->allocate(64u, ordinal);
                runtime_layers_ = static_cast<DeviceMoELayerRuntime *>(
                    backend_->allocate(runtime_bytes, ordinal));
                prepared_arrivals_ = static_cast<
                    DeviceMoEExpertDescriptor *>(
                    backend_->allocate(arrival_bytes, ordinal));
                apply_status_ = static_cast<
                    MoEOverlayDeviceRuntimeApplyStatus *>(
                    backend_->allocate(sizeof(*apply_status_), ordinal));
                arrival_staging_ = static_cast<
                    DeviceMoEExpertDescriptor *>(
                    backend_->allocatePinned(arrival_bytes, ordinal));
                transfer_stream_ = backend_->createStream(ordinal);
                transfer_event_ = backend_->createEvent(ordinal);
                epoch_arena_ = std::make_shared<DeviceMoEOverlayEpochArena>(
                    DeviceMoEOverlayEpochArena::Config{
                        .device_id = participant.device,
                        .initial_epoch = input.base_epoch,
                        .initial_bank = 0u,
                        // Two slots let the integration rig retain one reader
                        // from the retiring epoch while proving that a second
                        // request can acquire the newly admitted epoch.
                        .request_slot_capacity = 2u,
                        .external_admission_epoch = external_admission_epoch,
                        .external_admission_lifetime =
                            std::move(external_admission_lifetime),
                    });
                if (!dummy_payload_ || !dummy_scales_ || !runtime_layers_ ||
                    !prepared_arrivals_ || !apply_status_ ||
                    !arrival_staging_ || !transfer_stream_ ||
                    !transfer_event_ || !epoch_arena_)
                {
                    throw std::runtime_error(
                        "publication runtime fixture allocation failed");
                }

                std::vector<DeviceMoELayerRuntime> host_layers(
                    input.num_layers);
                for (std::uint32_t layer = 0u;
                     layer < input.num_layers;
                     ++layer)
                {
                    initializeLayer(
                        host_layers[layer], *group, input, layer);
                }

                if (!backend_->hostToDeviceOnStream(
                        runtime_layers_,
                        host_layers.data(),
                        runtime_bytes,
                        ordinal,
                        transfer_stream_) ||
                    !backend_->memset(
                        prepared_arrivals_,
                        0,
                        arrival_bytes,
                        ordinal,
                        transfer_stream_) ||
                    !backend_->memset(
                        apply_status_,
                        0,
                        sizeof(*apply_status_),
                        ordinal,
                        transfer_stream_) ||
                    !backend_->recordEvent(
                        transfer_event_, ordinal, transfer_stream_) ||
                    !awaitTransfer(std::chrono::seconds(5)))
                {
                    throw std::runtime_error(
                        "publication runtime fixture setup upload failed");
                }

                runtime_binding_ = {
                    .device = participant.device,
                    .runtime_layers_device = runtime_layers_,
                    .overlay_participant_id = participant.participant_id,
                    .domain_participant_id = static_cast<std::uint32_t>(
                        participant.domain_participant_index),
                    .domain_participant_count = static_cast<std::uint32_t>(
                        group->participant_ids.size()),
                    .layer_count = input.num_layers,
                    .expert_count = input.num_experts,
                    .top_k = 2u,
                    .epoch_control = epoch_arena_->control(),
                    .maintenance_epoch = epoch_arena_->maintenanceEpoch(),
                    .maintenance_status = epoch_arena_->maintenanceStatus(),
                };
                if (!runtime_binding_.publicationValid())
                {
                    throw std::logic_error(
                        "publication runtime fixture produced an invalid binding");
                }
            }
            catch (...)
            {
                release();
                throw;
            }
        }

        /** Drain only a live transfer event, then release model-lifetime state. */
        ~MoEOverlayDeviceRuntimePublicationFixture()
        {
            if (arrival_in_flight_ && transfer_event_)
            {
                (void)backend_->waitForEvent(
                    transfer_event_, participant_.device.ordinal);
            }
            release();
        }

        MoEOverlayDeviceRuntimePublicationFixture(
            const MoEOverlayDeviceRuntimePublicationFixture &) = delete;
        MoEOverlayDeviceRuntimePublicationFixture &operator=(
            const MoEOverlayDeviceRuntimePublicationFixture &) = delete;

        /** @return Immutable model runtime/epoch binding for this device. */
        [[nodiscard]] const MoEOverlayDeviceControllerRuntimeBinding &
        runtimeBinding() const noexcept
        {
            return runtime_binding_;
        }

        /** @return Complete action binding for the retained apply kernel. */
        [[nodiscard]] MoEOverlayDeviceRuntimePublicationBinding
        publicationBinding() const noexcept
        {
            return {
                .runtime_layers = runtime_layers_,
                .epoch_control = epoch_arena_->control(),
                .epoch_status = epoch_arena_->maintenanceStatus(),
                .prepared_arrivals = prepared_arrivals_,
                .apply_status = apply_status_,
                .arrival_capacity = arrival_capacity_,
                .layer_count = runtime_binding_.layer_count,
                .expert_count = runtime_binding_.expert_count,
                .domain_participant_id =
                    runtime_binding_.domain_participant_id,
                .domain_participant_count =
                    runtime_binding_.domain_participant_count,
            };
        }

        /**
         * @return Stable device ticket for one production RCU request slot.
         * @throws std::out_of_range when @p slot exceeds fixture capacity.
         */
        [[nodiscard]] DeviceMoEOverlayEpochTicket *requestTicket(
            std::uint32_t slot)
        {
            return epoch_arena_->requestTicket(slot);
        }

        /**
         * @return Stable device status for one production RCU request slot.
         * @throws std::out_of_range when @p slot exceeds fixture capacity.
         */
        [[nodiscard]] DeviceMoEOverlayEpochStatus *requestStatus(
            std::uint32_t slot)
        {
            return epoch_arena_->requestStatus(slot);
        }

        /** Copy one request status after its exact producer event is complete. */
        [[nodiscard]] bool copyRequestStatus(
            std::uint32_t slot,
            DeviceMoEOverlayEpochStatus *status)
        {
            if (!status)
                return false;
            return backend_->deviceToHost(
                status,
                epoch_arena_->requestStatus(slot),
                sizeof(*status),
                participant_.device.ordinal,
                transfer_stream_);
        }

        /** Copy terminal RCU control evidence after an exact event is complete. */
        [[nodiscard]] bool copyEpochControl(
            DeviceMoEOverlayEpochControl *control)
        {
            if (!control)
                return false;
            return backend_->deviceToHost(
                control,
                epoch_arena_->control(),
                sizeof(*control),
                participant_.device.ordinal,
                transfer_stream_);
        }

        /** Copy terminal maintenance status after its exact event is complete. */
        [[nodiscard]] bool copyMaintenanceStatus(
            DeviceMoEOverlayEpochStatus *status)
        {
            if (!status)
                return false;
            return backend_->deviceToHost(
                status,
                epoch_arena_->maintenanceStatus(),
                sizeof(*status),
                participant_.device.ordinal,
                transfer_stream_);
        }

        /**
         * @brief Publish a test-owned post-readiness acquisition guard value.
         * @param value Exact synthetic in-flight acquisition count.
         * @return True after the asynchronous upload's event completes.
         *
         * This adversarial hook models an inference admission that begins only
         * after the topology-wide old-bank readiness receipt. Such an
         * acquisition can select only the new epoch and must not invalidate
         * certified reclamation of the old bank. Production never writes this
         * device-owned word from the host; the hook exists solely to make that
         * narrow interleaving deterministic in the real-device protocol test.
         */
        [[nodiscard]] bool setPostReadinessAcquisitionGuardForTest(
            std::uint64_t value)
        {
            if (arrival_in_flight_)
                return false;
            const int ordinal = participant_.device.ordinal;
            auto *const guard = reinterpret_cast<std::uint64_t *>(
                reinterpret_cast<std::byte *>(epoch_arena_->control()) +
                offsetof(
                    DeviceMoEOverlayEpochControl,
                    acquisitions_in_flight));
            return backend_->hostToDeviceOnStream(
                       guard,
                       &value,
                       sizeof(value),
                       ordinal,
                       transfer_stream_) &&
                backend_->recordEvent(
                    transfer_event_, ordinal, transfer_stream_) &&
                awaitTransfer(std::chrono::seconds(5));
        }

        /**
         * @brief Enqueue exact destination arrivals without waiting.
         * @param batch Authenticated immutable controller command.
         * @param error Optional transfer-submission diagnostic.
         * @param omitted_ordinal Optional adversarial lane left empty while its
         *        real transfer event still completes.
         * @return False when the batch or transfer submission is invalid.
         */
        [[nodiscard]] bool enqueuePreparedArrivals(
            const MoEOverlayDeviceTransportCommandBatch &batch,
            std::string *error,
            std::optional<std::uint32_t> omitted_ordinal = std::nullopt)
        {
            if (arrival_in_flight_ || !batch.valid() ||
                batch.entries.size() > arrival_capacity_ ||
                (omitted_ordinal.has_value() &&
                 *omitted_ordinal >= batch.entries.size()))
            {
                if (error)
                    *error =
                        "arrival inbox received an invalid or overlapping batch";
                return false;
            }
            std::fill_n(
                arrival_staging_,
                arrival_capacity_,
                DeviceMoEExpertDescriptor{});
            for (const auto &entry : batch.entries)
            {
                if (entry.destination_participant !=
                    static_cast<std::uint32_t>(participant_.participant_id))
                {
                    continue;
                }
                if (omitted_ordinal.has_value() &&
                    entry.ordinal == *omitted_ordinal)
                {
                    // Adversarial integration tests leave one exact inbox lane
                    // empty while still completing its DMA/event. Production
                    // action code must reject that semantic omission before
                    // reserving or publishing a local RCU bank.
                    continue;
                }
                arrival_staging_[entry.ordinal] = descriptorFor(
                    entry.expert,
                    static_cast<std::int32_t>(entry.expert));
            }
            const int ordinal = participant_.device.ordinal;
            if (!backend_->hostToDeviceOnStream(
                    prepared_arrivals_,
                    arrival_staging_,
                    static_cast<std::size_t>(arrival_capacity_) *
                        sizeof(DeviceMoEExpertDescriptor),
                    ordinal,
                    transfer_stream_) ||
                !backend_->recordEvent(
                    transfer_event_, ordinal, transfer_stream_))
            {
                if (error)
                    *error =
                        "arrival inbox asynchronous upload was rejected";
                return false;
            }
            arrival_in_flight_ = true;
            return true;
        }

        /** @return Whether the exact participant transfer event completed. */
        [[nodiscard]] bool awaitPreparedArrivals(std::string *error)
        {
            if (!arrival_in_flight_ ||
                !awaitTransfer(std::chrono::seconds(5)))
            {
                if (error)
                    *error =
                        "arrival inbox transfer event did not complete";
                return false;
            }
            arrival_in_flight_ = false;
            return true;
        }

        /**
         * @brief Join the exact prepared-arrival DMA from a maintenance stream.
         *
         * The background worker must have submitted @ref enqueuePreparedArrivals
         * before this call, but the transfer need not be complete. This method
         * enqueues a device event dependency and returns immediately; it never
         * waits on either stream from the host.
         *
         * @param consumer_stream Exact participant maintenance stream.
         * @param error Optional rejection diagnostic.
         * @return True when the dependency was accepted by the owning backend.
         */
        [[nodiscard]] bool enqueuePreparedArrivalDependency(
            void *consumer_stream,
            std::string *error) const
        {
            const int ordinal = participant_.device.ordinal;
            if (!consumer_stream || !transfer_event_ ||
                !backend_->streamWaitEvent(
                    consumer_stream, transfer_event_, ordinal))
            {
                if (error)
                    *error =
                        "maintenance stream rejected its exact arrival event";
                return false;
            }
            return true;
        }

        /** Copy terminal apply evidence after the participant event completes. */
        [[nodiscard]] bool copyEvidence(
            MoEOverlayDeviceRuntimeApplyStatus *status,
            std::vector<DeviceMoELayerRuntime> *layers,
            DeviceMoEOverlayEpochControl *control = nullptr,
            DeviceMoEOverlayEpochStatus *epoch_status = nullptr)
        {
            if (!status || !layers)
                return false;
            layers->resize(runtime_binding_.layer_count);
            const int ordinal = participant_.device.ordinal;
            bool copied = backend_->deviceToHost(
                              status,
                              apply_status_,
                              sizeof(*status),
                              ordinal,
                              transfer_stream_) &&
                backend_->deviceToHost(
                    layers->data(),
                    runtime_layers_,
                    layers->size() * sizeof(DeviceMoELayerRuntime),
                    ordinal,
                    transfer_stream_);
            if (copied && control)
            {
                copied = backend_->deviceToHost(
                    control,
                    epoch_arena_->control(),
                    sizeof(*control),
                    ordinal,
                    transfer_stream_);
            }
            if (copied && epoch_status)
            {
                copied = backend_->deviceToHost(
                    epoch_status,
                    epoch_arena_->maintenanceStatus(),
                    sizeof(*epoch_status),
                    ordinal,
                    transfer_stream_);
            }
            return copied;
        }

        /** @return Global immutable identity of this local fixture. */
        [[nodiscard]] int participantId() const noexcept
        {
            return participant_.participant_id;
        }

    private:
        /** Poll the fixture's exact event without blocking its transfer stream. */
        [[nodiscard]] bool awaitTransfer(
            std::chrono::steady_clock::duration timeout) const
        {
            const auto deadline = std::chrono::steady_clock::now() + timeout;
            while (std::chrono::steady_clock::now() < deadline)
            {
                bool ready = false;
                if (!backend_->queryEvent(
                        transfer_event_, participant_.device.ordinal, &ready))
                {
                    return false;
                }
                if (ready)
                    return true;
                std::this_thread::yield();
            }
            return false;
        }

        /** Resolve the unique authoritative owner from one packed snapshot. */
        [[nodiscard]] static std::uint32_t ownerFor(
            const MoEOverlayDevicePlacementPolicyInput &input,
            std::uint32_t layer,
            std::uint32_t expert)
        {
            std::uint32_t owner =
                static_cast<std::uint32_t>(input.participants.size());
            for (std::uint32_t participant = 0u;
                 participant < input.participants.size();
                 ++participant)
            {
                const std::size_t index =
                    (static_cast<std::size_t>(participant) * input.num_layers +
                     layer) *
                        input.num_experts +
                    expert;
                if (moe_rebalance_policy::collectedStateAuthoritativeOwner(
                        input.collected_state.at(index)))
                {
                    if (owner != input.participants.size())
                    {
                        throw std::logic_error(
                            "snapshot has multiple authoritative owners");
                    }
                    owner = participant;
                }
            }
            if (owner == input.participants.size())
                throw std::logic_error("snapshot has no authoritative owner");
            return owner;
        }

        /** Initialize one layer without inventing a participant-local owner map. */
        void initializeLayer(
            DeviceMoELayerRuntime &runtime,
            const MoEOverlayDeviceControllerGroup &local_group,
            const MoEOverlayDevicePlacementPolicyInput &input,
            std::uint32_t layer) const
        {
            runtime.active_bank = 0u;
            runtime.active_epoch =
                static_cast<std::uint32_t>(input.base_epoch);
            runtime.expert_count = input.num_experts;
            runtime.top_k = 2u;
            runtime.participant_id = static_cast<std::uint32_t>(
                participant_.domain_participant_index);
            runtime.participant_count = static_cast<std::uint32_t>(
                local_group.participant_ids.size());
            auto &bank = runtime.banks[0];
            bank.epoch = static_cast<std::uint32_t>(input.base_epoch);
            bank.expert_count = input.num_experts;
            for (std::uint32_t expert = 0u;
                 expert < input.num_experts;
                 ++expert)
            {
                const std::uint32_t owner = ownerFor(input, layer, expert);
                const auto &owner_metadata = topology_->participants.at(owner);
                const auto *const owner_group =
                    topology_->groupForParticipant(static_cast<int>(owner));
                if (!owner_group)
                    throw std::logic_error("base owner has no controller group");

                bank.overlay_route_participant[expert] =
                    static_cast<std::int32_t>(owner);
                auto &descriptor = bank.experts[expert];
                descriptor.logical_expert_id =
                    static_cast<std::int32_t>(expert);
                descriptor.local_slot = -1;
                if (owner_group->group_id == local_group.group_id)
                {
                    const auto local_owner = static_cast<std::uint32_t>(
                        owner_metadata.domain_participant_index);
                    descriptor.owner_participant =
                        static_cast<std::int32_t>(local_owner);
                    bank.resident_participant_mask[expert] =
                        1u << local_owner;
                    if (owner == static_cast<std::uint32_t>(
                                     participant_.participant_id))
                    {
                        descriptor = descriptorFor(
                            expert, static_cast<std::int32_t>(expert));
                        descriptor.owner_participant =
                            static_cast<std::int32_t>(local_owner);
                        descriptor.flags = toMoEExpertFlags(
                            DeviceMoEExpertFlags::Valid |
                            DeviceMoEExpertFlags::Resident |
                            DeviceMoEExpertFlags::PreferredOwner |
                            DeviceMoEExpertFlags::LocalCompute);
                        bank.local_compute_mask[expert] = 1u;
                        bank.replica_role[expert] = static_cast<std::uint8_t>(
                            DeviceMoEReplicaRole::Primary);
                    }
                }
                else
                {
                    descriptor.owner_participant = -1;
                }

                const std::size_t state_index =
                    (static_cast<std::size_t>(participant_.participant_id) *
                         input.num_layers +
                     layer) *
                        input.num_experts +
                    expert;
                runtime.decode_histogram[expert] =
                    moe_rebalance_policy::collectedStateActivationCount(
                        input.collected_state.at(state_index));
                runtime.decode_local_histogram[expert] =
                    runtime.decode_histogram[expert];
            }
        }

        /**
         * @brief Build one complete durable physical-ledger descriptor.
         *
         * The fixture models the production ExpertOverlay arrival inbox, not
         * request-scoped LLEP storage. The descriptor therefore follows the
         * ordinary logical expert-slot convention and must not claim a
         * `DeviceMoETransferSlotDirectory` allocation.
         */
        [[nodiscard]] DeviceMoEExpertDescriptor descriptorFor(
            std::uint32_t expert,
            std::int32_t local_slot) const
        {
            DeviceMoEExpertDescriptor descriptor;
            const DeviceNativeVNNIMatrixDesc gate_up{
                .payload =
                    static_cast<const std::uint8_t *>(dummy_payload_),
                .scales = dummy_scales_,
                .n = 4,
                .k = 32,
                .blocks_per_row = 1u,
            };
            const DeviceNativeVNNIMatrixDesc down{
                .payload =
                    static_cast<const std::uint8_t *>(dummy_payload_),
                .scales = dummy_scales_,
                .n = 32,
                .k = 4,
                .blocks_per_row = 1u,
            };
            descriptor.gate = gate_up;
            descriptor.up = gate_up;
            descriptor.down = down;
            descriptor.logical_expert_id =
                static_cast<std::int32_t>(expert);
            descriptor.local_slot = local_slot;
            descriptor.flags = toMoEExpertFlags(
                DeviceMoEExpertFlags::Valid |
                DeviceMoEExpertFlags::Resident);
            return descriptor;
        }

        /** Release every partially or fully constructed resource. */
        void release() noexcept
        {
            epoch_arena_.reset();
            if (!backend_)
                return;
            const int ordinal = participant_.device.ordinal;
            if (transfer_event_)
                backend_->destroyEvent(transfer_event_, ordinal);
            if (transfer_stream_)
                backend_->destroyStream(transfer_stream_, ordinal);
            if (arrival_staging_)
                backend_->freePinned(arrival_staging_, ordinal);
            if (apply_status_)
                backend_->free(apply_status_, ordinal);
            if (prepared_arrivals_)
                backend_->free(prepared_arrivals_, ordinal);
            if (runtime_layers_)
                backend_->free(runtime_layers_, ordinal);
            if (dummy_scales_)
                backend_->free(dummy_scales_, ordinal);
            if (dummy_payload_)
                backend_->free(dummy_payload_, ordinal);
            transfer_event_ = nullptr;
            transfer_stream_ = nullptr;
            arrival_staging_ = nullptr;
            apply_status_ = nullptr;
            prepared_arrivals_ = nullptr;
            runtime_layers_ = nullptr;
            dummy_scales_ = nullptr;
            dummy_payload_ = nullptr;
        }

        IBackend *backend_ = nullptr;
        MoEExpertOwnerParticipant participant_;
        const MoEOverlayDeviceControllerTopology *topology_ = nullptr;
        std::uint32_t arrival_capacity_ = 0u;
        void *dummy_payload_ = nullptr;
        void *dummy_scales_ = nullptr;
        DeviceMoELayerRuntime *runtime_layers_ = nullptr;
        DeviceMoEExpertDescriptor *prepared_arrivals_ = nullptr;
        MoEOverlayDeviceRuntimeApplyStatus *apply_status_ = nullptr;
        DeviceMoEExpertDescriptor *arrival_staging_ = nullptr;
        void *transfer_stream_ = nullptr;
        void *transfer_event_ = nullptr;
        bool arrival_in_flight_ = false;
        std::shared_ptr<DeviceMoEOverlayEpochArena> epoch_arena_;
        MoEOverlayDeviceControllerRuntimeBinding runtime_binding_;
    };
} // namespace llaminar2::test
