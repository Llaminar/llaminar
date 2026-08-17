/**
 * @file DeviceMoEOverlayEpochArena.h
 * @brief Model-lifetime storage for captured ExpertOverlay epoch transactions.
 */

#pragma once

#include "DeviceMoEOverlayEpochABI.h"

#include "../../backends/DeviceId.h"

#include <cstddef>
#include <cstdint>

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

        /** @return Stable semantic status for serialized maintenance operations. */
        [[nodiscard]] DeviceMoEOverlayEpochStatus *maintenanceStatus() noexcept
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
    };
} // namespace llaminar2
