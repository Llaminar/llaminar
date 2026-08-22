/**
 * @file MoEOverlayDevicePreparedArrivalInbox.h
 * @brief Persistent descriptor publication for device-authored MoE movement.
 *
 * Physical migration prepares gate/up/down engines asynchronously in inactive
 * slots.  The device controller subsequently needs the six pointer-bearing
 * matrix descriptors at stable addresses while it builds a new runtime bank.
 * This class owns that narrow handoff: one preallocated device array, one
 * pinned staging array, one exact transfer stream, one completion event, and
 * one device-side semantic status record per participant.
 *
 * No placement policy lives here.  The immutable device command determines
 * descriptor ordinals and destinations; the completed physical wave supplies
 * the retained engine lifetimes. These durable physical-ledger allocations are
 * deliberately distinct from request-scoped `DeviceMoETransferSlotDirectory`
 * storage. Runtime waves allocate nothing, never use a default stream, and
 * expose only event-based ordering to the controller.
 */

#pragma once

#include "MoEOverlayDeviceControllerKernels.h"
#include "MoEOverlayDeviceControllerRuntimeBinding.h"
#include "MoEOverlayDevicePhysicalMovement.h"
#include "MoEOverlayParticipantMigration.h"

#include <cstdint>
#include <string>

namespace llaminar2
{
    class IBackend;

    /**
     * @brief Model-lifetime arrival descriptor inbox for one GPU participant.
     *
     * Construction and destruction must execute on the owning device worker.
     * Runtime methods are serialized by the controller service: at most one
     * descriptor DMA may be live, and a new wave cannot begin until the prior
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
            /** Exact controller stream that consumes every published wave. */
            void *consumer_stream = nullptr;
            std::string perf_device;
        };

        /**
         * @brief Allocate and asynchronously initialize the persistent inbox.
         *
         * Initialization is submitted on the inbox transfer stream and joined
         * to `Config::consumer_stream` with an event edge. Construction never
         * waits on the host; subsequent uploads remain ordered after the same
         * initialization because they use the producer stream in FIFO order.
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
         * @brief Export and asynchronously upload this participant's arrivals.
         *
         * `prepared` covers every destination owned by the current process.
         * This inbox exports only its exact participant and deliberately skips
         * non-null lifetimes retained for sibling devices on the same rank.
         * The corresponding command lanes stay zero in this device's array.
         *
         * @param batch Exact physical projection of the device command.
         * @param prepared Completed projection operations in command order.
         * @param error Optional exact rejection diagnostic.
         * @return True after the H2D and its event have been enqueued.
         */
        [[nodiscard]] bool enqueue(
            const MoEOverlayDevicePhysicalMovementBatch &batch,
            const MoEOverlayParticipantPreparedTransfers &prepared,
            std::string *error = nullptr) noexcept;

        /**
         * @brief Order the configured controller stream after descriptor DMA.
         *
         * The consumer stream is fixed at construction, preventing a wave from
         * accidentally publishing its readiness edge to an unrelated stream.
         *
         * @param error Optional exact rejection diagnostic.
         * @return True when the backend accepted the event dependency.
         */
        [[nodiscard]] bool enqueueDependency(
            std::string *error = nullptr) const noexcept;

        /**
         * @brief Poll descriptor DMA completion without synchronizing a stream.
         * @param ready Receives true only after the exact event completes.
         * @param error Optional backend rejection diagnostic.
         * @return True when the event query itself succeeded.
         */
        [[nodiscard]] bool queryReady(
            bool *ready,
            std::string *error = nullptr) const noexcept;

        /**
         * @brief Mark a completed controller transaction reusable.
         * @param error Optional lifecycle rejection diagnostic.
         * @return True only after the descriptor event is known complete.
         */
        [[nodiscard]] bool finishWave(
            std::string *error = nullptr) noexcept;

        /** @return Exact participant id whose destination lanes are accepted. */
        [[nodiscard]] int participantId() const noexcept
        {
            return config_.runtime_binding.overlay_participant_id;
        }

        /** @return Dedicated producer stream used only by descriptor uploads. */
        [[nodiscard]] void *transferStream() const noexcept
        {
            return transfer_stream_;
        }

    private:
        /** Complete lifecycle of the single reusable producer event. */
        enum class State : std::uint8_t
        {
            InitializationSubmitted, ///< Initial zeroing is ordered, not polled.
            Ready,                   ///< No descriptor upload remains in flight.
            WaveSubmitted,           ///< One descriptor upload owns the event.
            Released,                ///< All device and host storage is retired.
        };

        /** Release any partially materialized setup resources. */
        void release() noexcept;

        Config config_;
        DeviceMoEExpertDescriptor *prepared_arrivals_device_ = nullptr;
        DeviceMoEExpertDescriptor *prepared_arrivals_staging_ = nullptr;
        MoEOverlayDeviceRuntimeApplyStatus *apply_status_device_ = nullptr;
        void *transfer_stream_ = nullptr;
        void *transfer_event_ = nullptr;
        State state_ = State::InitializationSubmitted;
    };
} // namespace llaminar2
