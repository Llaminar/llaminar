/**
 * @file MoEOverlayDevicePreparedArrivalInbox.h
 * @brief Persistent descriptor publication for device-authored MoE movement.
 *
 * Physical migration prepares gate/up/down engines asynchronously in inactive
 * slots.  The device controller subsequently needs the six pointer-bearing
 * matrix descriptors at stable addresses while it builds a new runtime bank.
 * This class owns that narrow handoff: one TransferEngine-owned mapped array
 * and one device-side semantic status record per participant. The transport
 * receipt releases immutable host-prepared descriptors to the captured device
 * candidate builder, which copies them into its inactive device runtime bank.
 *
 * No placement policy lives here.  The immutable device command determines
 * descriptor ordinals and destinations; the completed physical wave supplies
 * the retained engine lifetimes. These durable physical-ledger allocations are
 * deliberately distinct from request-scoped `DeviceMoETransferSlotDirectory`
 * storage. Runtime staging allocates nothing and submits no GPU work. In
 * particular it cannot wait behind inference on a copy-engine work queue.
 */

#pragma once

#include "MoEOverlayDeviceControllerKernels.h"
#include "MoEOverlayDeviceControllerRuntimeBinding.h"
#include "MoEOverlayDevicePhysicalMovement.h"
#include "MoEOverlayParticipantMigration.h"

#include <cstdint>
#include <memory>
#include <string>

namespace llaminar2
{
    class IBackend;
    class MappedHostTransferRegion;
    class MoEOverlayDeviceTransportProtocol;

    /**
     * @brief Model-lifetime arrival descriptor inbox for one GPU participant.
     *
     * Construction and destruction must execute on the owning device worker.
     * Runtime methods are serialized by the controller service: at most one
     * immutable publication may be live, and a new wave cannot begin until the prior
     * device transaction reaches Complete.  The class does not own physical
     * weight storage; those lifetimes remain in the physical slot ledger.
     */
    class MoEOverlayDevicePreparedArrivalInbox final
    {
    public:
        /** @brief Stable participant and preallocated command geometry. */
        struct Config
        {
            IBackend *backend = nullptr;
            MoEOverlayDeviceControllerRuntimeBinding runtime_binding;
            std::uint32_t command_capacity = 0u;
            /** Exact controller stream used for setup-only status initialization. */
            void *consumer_stream = nullptr;
            std::string perf_device;
        };

        /**
         * @brief Allocate the mapped inbox and initialize device status at setup.
         *
         * Status zeroing precedes controller capture on the exact consumer
         * stream. Descriptor pages are initialized on the host. There is no
         * separate upload stream, device mirror, or reusable DMA event.
         *
         * @throws std::invalid_argument for incomplete publication geometry.
         * @throws std::runtime_error when setup allocation or zeroing fails.
         */
        explicit MoEOverlayDevicePreparedArrivalInbox(Config config);

        /** @brief Release setup-owned resources after the owning stream is idle. */
        ~MoEOverlayDevicePreparedArrivalInbox();

        MoEOverlayDevicePreparedArrivalInbox(
            const MoEOverlayDevicePreparedArrivalInbox &) = delete;
        MoEOverlayDevicePreparedArrivalInbox &operator=(
            const MoEOverlayDevicePreparedArrivalInbox &) = delete;

        /** @return Immutable-address binding embedded into controller graphs. */
        [[nodiscard]] MoEOverlayDeviceRuntimePublicationBinding
        publicationBinding() const noexcept;

        /**
         * @brief Stage immutable descriptors for this participant's arrivals.
         *
         * `prepared` covers every destination owned by the current process.
         * This inbox exports only its exact participant and deliberately skips
         * non-null lifetimes retained for sibling devices on the same rank.
         * The corresponding command lanes stay zero in this participant's array.
         * The caller then publishes the transport protocol's prepared receipt;
         * that system release/acquire edge covers these writes. Only the device
         * controller can turn these physical addresses into live expert state.
         *
         * @param batch Exact physical projection of the device command.
         * @param prepared Completed projection operations in command order.
         * @param error Optional exact rejection diagnostic.
         * @return True after staging; no GPU submission or wait occurs.
         */
        [[nodiscard]] bool stage(
            const MoEOverlayDevicePhysicalMovementBatch &batch,
            const MoEOverlayParticipantPreparedTransfers &prepared,
            std::string *error = nullptr) noexcept;

        /**
         * @brief Retire this publication only after its exact device completion.
         * @param protocol Read-only observer of the authoritative lifecycle.
         * @param command Exact immutable command whose descriptors are staged.
         * @param error Optional lifecycle rejection diagnostic.
         * @return True only after the controller completed this transaction.
         */
        [[nodiscard]] bool finishWave(
            const MoEOverlayDeviceTransportProtocol &protocol,
            const MoEOverlayDeviceTransportCommandBatch &command,
            std::string *error = nullptr) noexcept;

        /** @return Exact participant id whose destination lanes are accepted. */
        [[nodiscard]] int participantId() const noexcept
        {
            return config_.runtime_binding.overlay_participant_id;
        }

    private:
        /** Host-owned physical publication lifetime; never device policy state. */
        enum class State : std::uint8_t
        {
            Ready,                   ///< No device transaction borrows the pages.
            Staged,                  ///< Exact transaction owns immutable pages.
            Released,                ///< All device and host storage is retired.
        };

        /** Release any partially materialized setup resources. */
        void release() noexcept;

        Config config_;
        std::shared_ptr<MappedHostTransferRegion> mapped_arrivals_;
        DeviceMoEExpertDescriptor *prepared_arrivals_device_ = nullptr;
        DeviceMoEExpertDescriptor *prepared_arrivals_staging_ = nullptr;
        MoEOverlayDeviceRuntimeApplyStatus *apply_status_device_ = nullptr;
        std::uint64_t staged_transaction_ = 0u;
        std::uint64_t staged_digest_ = 0u;
        std::uint64_t retired_transaction_ = 0u;
        State state_ = State::Ready;
    };
} // namespace llaminar2
