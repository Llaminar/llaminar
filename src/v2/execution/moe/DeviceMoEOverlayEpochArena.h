/**
 * @file DeviceMoEOverlayEpochArena.h
 * @brief Model-lifetime storage for captured ExpertOverlay epoch transactions.
 */

#pragma once

#include "DeviceMoEOverlayEpochABI.h"

#include "../../backends/DeviceId.h"

#include <cstddef>
#include <cstdint>
#include <memory>

namespace llaminar2
{
    class IBackend;

    /**
     * @brief Own stable CPU or GPU addresses for placement-epoch RCU.
     *
     * One arena belongs to one physical participant and one serial request
     * family. Its control block is the publication authority on that device;
     * request slots retain reader-holding tickets across every captured model
     * and MTP graph that participates in the same transaction. Maintenance has
     * a separate epoch scalar and status record so background publication can
     * overlap inference without rebinding any captured address.
     *
     * Construction and destruction are model-lifecycle operations. No method
     * allocates, copies, or synchronizes after construction.
     */
    class DeviceMoEOverlayEpochArena final
    {
    public:
        /** @brief Complete immutable allocation and initial-publication policy. */
        struct Config
        {
            DeviceId device_id = DeviceId::invalid(); ///< Exact owning participant.
            std::uint64_t initial_epoch = 1u;         ///< First executable residency epoch.
            std::uint32_t initial_bank = 0u;          ///< Bank already populated for that epoch.
            std::uint32_t request_slot_capacity = 1u; ///< Maximum concurrent request tickets.
            /**
             * Optional system-visible topology-wide admission epoch.
             *
             * Multi-group device-resident ExpertOverlay publishes each
             * participant's local RCU bank independently, then advances one
             * mapped global admission word after every participant is ready.
             * Binding that word here makes request admission select the exact
             * globally approved epoch, including a still-readable retiring
             * local bank during the short publication fan-out window. Null
             * retains ordinary participant-local selector admission.
             */
            const std::uint64_t *external_admission_epoch = nullptr;
            /** Keeps the mapped admission allocation registered and alive. */
            std::shared_ptr<const void> external_admission_lifetime;
            /**
             * Optional two-phase continuation barrier for symmetric LocalTP.
             *
             * The binding is legal only beside `external_admission_epoch` and
             * uses the same model-lifetime mapped allocation. Remote segmented
             * followers instead receive an exact dispatch-selected epoch and do
             * not participate in this continuation-only barrier.
             */
            DeviceMoEOverlayEpochAdmissionBarrierBinding admission_barrier;
        };

        /**
         * @brief Allocate and initialize every stable epoch address.
         * @param config Immutable device, initial epoch, bank, and capacity.
         * @throws std::invalid_argument for malformed configuration.
         * @throws std::runtime_error when backend allocation or setup upload fails.
         */
        explicit DeviceMoEOverlayEpochArena(Config config);

        /** @brief Release model-lifetime epoch storage after all graphs drain. */
        ~DeviceMoEOverlayEpochArena();

        DeviceMoEOverlayEpochArena(
            const DeviceMoEOverlayEpochArena &) = delete;
        DeviceMoEOverlayEpochArena &operator=(
            const DeviceMoEOverlayEpochArena &) = delete;
        DeviceMoEOverlayEpochArena(
            DeviceMoEOverlayEpochArena &&) = delete;
        DeviceMoEOverlayEpochArena &operator=(
            DeviceMoEOverlayEpochArena &&) = delete;

        /** @return Exact CPU/GPU participant that owns every returned pointer. */
        [[nodiscard]] const DeviceId &deviceId() const noexcept
        {
            return device_id_;
        }

        /** @return Number of independently reader-holding request slots. */
        [[nodiscard]] std::uint32_t requestSlotCapacity() const noexcept
        {
            return request_slot_capacity_;
        }

        /** @return Authoritative backend-resident publication control block. */
        [[nodiscard]] DeviceMoEOverlayEpochControl *control() noexcept
        {
            return control_;
        }

        /** @return Const view of the authoritative publication control block. */
        [[nodiscard]] const DeviceMoEOverlayEpochControl *control() const noexcept
        {
            return control_;
        }

        /**
         * @return Stable mutable device address of the publication control block.
         *
         * This accessor is intentionally const with respect to the host arena:
         * resolving a retained graph binding neither changes allocation
         * topology nor mutates host-owned state.  Only kernels executing on
         * @ref deviceId may mutate the pointee, on their exact maintenance
         * stream.  Host code must not dereference or edit the returned block.
         */
        [[nodiscard]] DeviceMoEOverlayEpochControl *
        deviceControlAddress() const noexcept
        {
            return control_;
        }

        /**
         * @return Optional system-visible epoch that gates request admission.
         *
         * The pointer is an immutable model-topology address. Its value is
         * device-owned and must be consumed only by the captured acquire
         * kernel; host code must not dereference it as a placement mirror.
         */
        [[nodiscard]] const std::uint64_t *externalAdmissionEpoch() const noexcept
        {
            return external_admission_epoch_;
        }

        /**
         * @return Optional node-local exact-epoch barrier embedded by acquire.
         *
         * The returned value contains device aliases only. Host code may compare
         * their immutable identities but must not dereference the mapped record.
         */
        [[nodiscard]] DeviceMoEOverlayEpochAdmissionBarrierBinding
        admissionBarrier() const noexcept
        {
            return admission_barrier_;
        }

        /**
         * @brief Return the stable request ticket for one admitted slot.
         * @throws std::out_of_range when @p slot is outside the immutable capacity.
         */
        [[nodiscard]] DeviceMoEOverlayEpochTicket *requestTicket(
            std::uint32_t slot);

        /**
         * @brief Return the stable semantic status for one request slot.
         * @throws std::out_of_range when @p slot is outside the immutable capacity.
         */
        [[nodiscard]] DeviceMoEOverlayEpochStatus *requestStatus(
            std::uint32_t slot);

        /**
         * @brief Return the stable request-status address through a const arena.
         *
         * The arena's allocation topology is immutable; only captured device
         * kernels mutate the pointee on their exact stream.
         */
        [[nodiscard]] const DeviceMoEOverlayEpochStatus *requestStatus(
            std::uint32_t slot) const;

        /**
         * @return Device-owned next candidate epoch consumed by maintenance.
         *
         * Construction initializes it to `initial_epoch + 1`; a successful
         * captured publication advances it in place. Failed/busy/no-work
         * attempts retain the value for a later replay.
         */
        [[nodiscard]] std::uint64_t *maintenanceEpoch() noexcept
        {
            return maintenance_epoch_;
        }

        /**
         * @return Stable mutable device address of the maintenance epoch.
         *
         * Binding this address is a logically const model-topology lookup.
         * Mutation remains device-owned and ordered by the maintenance stream;
         * the host must never use this pointer as an epoch mirror.
         */
        [[nodiscard]] std::uint64_t *
        deviceMaintenanceEpochAddress() const noexcept
        {
            return maintenance_epoch_;
        }

        /** @return Stable semantic status for serialized maintenance operations. */
        [[nodiscard]] DeviceMoEOverlayEpochStatus *maintenanceStatus() noexcept
        {
            return maintenance_status_;
        }

        /**
         * @return Stable mutable device address of maintenance status storage.
         *
         * The address is immutable model topology even though the device
         * controller writes the pointee.  This distinction lets const graph
         * builders bind retained maintenance graphs without a const_cast.
         */
        [[nodiscard]] DeviceMoEOverlayEpochStatus *
        deviceMaintenanceStatusAddress() const noexcept
        {
            return maintenance_status_;
        }

        /**
         * @return Requested payload bytes owned by this arena.
         *
         * Allocator metadata and alignment padding are deliberately excluded so
         * the value can enter the model memory bill of materials.
         */
        [[nodiscard]] std::size_t allocationBytes() const noexcept;

    private:
        /** @brief Release a partially or fully constructed arena without throwing. */
        void release() noexcept;

        /** @brief Validate and return one request-slot array index. */
        [[nodiscard]] std::size_t checkedSlot(std::uint32_t slot) const;

        DeviceId device_id_ = DeviceId::invalid();
        std::uint32_t request_slot_capacity_ = 0u;
        IBackend *backend_ = nullptr;
        DeviceMoEOverlayEpochControl *control_ = nullptr;
        DeviceMoEOverlayEpochTicket *request_tickets_ = nullptr;
        DeviceMoEOverlayEpochStatus *request_statuses_ = nullptr;
        std::uint64_t *maintenance_epoch_ = nullptr;
        DeviceMoEOverlayEpochStatus *maintenance_status_ = nullptr;
        const std::uint64_t *external_admission_epoch_ = nullptr;
        std::shared_ptr<const void> external_admission_lifetime_;
        DeviceMoEOverlayEpochAdmissionBarrierBinding admission_barrier_;
    };
} // namespace llaminar2
