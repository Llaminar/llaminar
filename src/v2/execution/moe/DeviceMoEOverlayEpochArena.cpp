/**
 * @file DeviceMoEOverlayEpochArena.cpp
 * @brief Stable allocation and setup publication for ExpertOverlay epoch RCU.
 */

#include "DeviceMoEOverlayEpochArena.h"

#include "MoEOverlayDeviceEpochProtocol.h"
#include "../../backends/BackendManager.h"
#include "../../backends/IBackend.h"
#include "../../utils/Logger.h"
#include "../../utils/PerfStatsCollector.h"

#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace llaminar2
{
    namespace
    {
        /** @return Checked byte count for an array owned by the arena. */
        template <typename T>
        std::size_t checkedArrayBytes(
            std::uint32_t count,
            const char *label)
        {
            if (count > std::numeric_limits<std::size_t>::max() / sizeof(T))
            {
                throw std::overflow_error(
                    std::string("ExpertOverlay epoch arena ") + label +
                    " byte count overflows size_t");
            }
            return static_cast<std::size_t>(count) * sizeof(T);
        }

        /**
         * @brief Model-setup stream that makes GPU initialization explicit.
         *
         * The stream never enters inference or capture. Its destructor only
         * destroys the handle; callers must complete setup work successfully
         * before leaving the constructor.
         */
        class SetupStream final
        {
        public:
            /** @brief Create one exact non-default setup stream. */
            SetupStream(IBackend &backend, int ordinal)
                : backend_(&backend), ordinal_(ordinal), stream_(backend.createStream(ordinal))
            {
                if (!stream_)
                {
                    throw std::runtime_error(
                        "ExpertOverlay epoch arena could not create its GPU setup stream");
                }
            }

            /** @brief Destroy the setup stream after its initialization fence. */
            ~SetupStream()
            {
                if (backend_ && stream_)
                    backend_->destroyStream(stream_, ordinal_);
            }

            SetupStream(const SetupStream &) = delete;
            SetupStream &operator=(const SetupStream &) = delete;

            /** @return Opaque exact stream passed to backend setup copies. */
            [[nodiscard]] void *get() const noexcept { return stream_; }

        private:
            IBackend *backend_ = nullptr;
            int ordinal_ = -1;
            void *stream_ = nullptr;
        };

        /**
         * @brief Upload one initialized model-lifetime object without default streams.
         */
        void uploadSetupObject(
            IBackend &backend,
            int ordinal,
            void *stream,
            void *destination,
            const void *source,
            std::size_t bytes,
            const char *label)
        {
            if (!destination || !source || bytes == 0u || !stream ||
                !backend.hostToDeviceOnStream(
                    destination, source, bytes, ordinal, stream))
            {
                throw std::runtime_error(
                    std::string("ExpertOverlay epoch arena failed to upload ") +
                    label);
            }
        }
    } // namespace

    DeviceMoEOverlayEpochArena::DeviceMoEOverlayEpochArena(Config config)
        : device_id_(config.device_id),
          request_slot_capacity_(config.request_slot_capacity)
    {
        if (!device_id_.is_valid())
        {
            throw std::invalid_argument(
                "ExpertOverlay epoch arena requires a valid device");
        }
        if (config.initial_epoch == 0u)
        {
            throw std::invalid_argument(
                "ExpertOverlay epoch arena requires a positive initial epoch");
        }
        if (config.initial_epoch ==
            std::numeric_limits<std::uint64_t>::max())
        {
            throw std::invalid_argument(
                "ExpertOverlay epoch arena initial epoch leaves no maintenance generation");
        }
        if (config.initial_bank >= kDeviceMoEOverlayEpochBankCount)
        {
            throw std::invalid_argument(
                "ExpertOverlay epoch arena initial bank is outside the two-bank ABI");
        }
        if (request_slot_capacity_ == 0u)
        {
            throw std::invalid_argument(
                "ExpertOverlay epoch arena requires at least one request slot");
        }

        const std::size_t ticket_bytes =
            checkedArrayBytes<DeviceMoEOverlayEpochTicket>(
                request_slot_capacity_, "ticket");
        const std::size_t status_bytes =
            checkedArrayBytes<DeviceMoEOverlayEpochStatus>(
                request_slot_capacity_, "status");

        DeviceMoEOverlayEpochControl initial_control{};
        MoEOverlayDeviceEpochProtocol::initialize(
            initial_control, config.initial_epoch, config.initial_bank);
        std::vector<DeviceMoEOverlayEpochTicket> initial_tickets(
            request_slot_capacity_);
        std::vector<DeviceMoEOverlayEpochStatus> initial_statuses(
            request_slot_capacity_);
        /* The captured maintenance graph owns this scalar. It reuses the same
         * address on every replay and advances it only after a successful
         * publication, so no host shadow or recapture is required. */
        const std::uint64_t initial_maintenance_epoch =
            config.initial_epoch + 1u;
        DeviceMoEOverlayEpochStatus initial_maintenance_status{};

        try
        {
            if (device_id_.is_cpu())
            {
                control_ = new DeviceMoEOverlayEpochControl(initial_control);
                request_tickets_ = new DeviceMoEOverlayEpochTicket[
                    request_slot_capacity_]{};
                request_statuses_ = new DeviceMoEOverlayEpochStatus[
                    request_slot_capacity_]{};
                maintenance_epoch_ = new std::uint64_t(
                    initial_maintenance_epoch);
                maintenance_status_ = new DeviceMoEOverlayEpochStatus{};
            }
            else
            {
                backend_ = getBackendFor(device_id_);
                if (!backend_)
                {
                    throw std::runtime_error(
                        "ExpertOverlay epoch arena has no backend for " +
                        device_id_.to_string());
                }
                const int ordinal = device_id_.toKernelDeviceIndex();
                control_ = static_cast<DeviceMoEOverlayEpochControl *>(
                    backend_->allocate(sizeof(*control_), ordinal));
                request_tickets_ =
                    static_cast<DeviceMoEOverlayEpochTicket *>(
                        backend_->allocate(ticket_bytes, ordinal));
                request_statuses_ =
                    static_cast<DeviceMoEOverlayEpochStatus *>(
                        backend_->allocate(status_bytes, ordinal));
                maintenance_epoch_ = static_cast<std::uint64_t *>(
                    backend_->allocate(sizeof(*maintenance_epoch_), ordinal));
                maintenance_status_ =
                    static_cast<DeviceMoEOverlayEpochStatus *>(
                        backend_->allocate(
                            sizeof(*maintenance_status_), ordinal));
                if (!control_ || !request_tickets_ || !request_statuses_ ||
                    !maintenance_epoch_ || !maintenance_status_)
                {
                    throw std::runtime_error(
                        "ExpertOverlay epoch arena backend allocation failed on " +
                        device_id_.to_string());
                }

                SetupStream setup(*backend_, ordinal);
                uploadSetupObject(
                    *backend_, ordinal, setup.get(), control_,
                    &initial_control, sizeof(initial_control), "control");
                uploadSetupObject(
                    *backend_, ordinal, setup.get(), request_tickets_,
                    initial_tickets.data(), ticket_bytes, "request tickets");
                uploadSetupObject(
                    *backend_, ordinal, setup.get(), request_statuses_,
                    initial_statuses.data(), status_bytes, "request statuses");
                uploadSetupObject(
                    *backend_, ordinal, setup.get(), maintenance_epoch_,
                    &initial_maintenance_epoch,
                    sizeof(initial_maintenance_epoch), "maintenance epoch");
                uploadSetupObject(
                    *backend_, ordinal, setup.get(), maintenance_status_,
                    &initial_maintenance_status,
                    sizeof(initial_maintenance_status), "maintenance status");
                if (!backend_->synchronizeStream(setup.get(), ordinal))
                {
                    throw std::runtime_error(
                        "ExpertOverlay epoch arena setup stream did not complete on " +
                        device_id_.to_string());
                }
            }
        }
        catch (...)
        {
            release();
            throw;
        }

        PerfStatsCollector::addCounter(
            "memory",
            "moe_overlay_epoch_arena_allocations",
            1.0,
            "model_setup",
            device_id_.to_string(),
            {{"bytes", std::to_string(allocationBytes())},
             {"request_slots", std::to_string(request_slot_capacity_)},
             {"initial_epoch", std::to_string(config.initial_epoch)},
             {"initial_bank", std::to_string(config.initial_bank)}});
    }

    DeviceMoEOverlayEpochArena::~DeviceMoEOverlayEpochArena()
    {
        release();
    }

    std::size_t DeviceMoEOverlayEpochArena::checkedSlot(
        std::uint32_t slot) const
    {
        if (slot >= request_slot_capacity_)
        {
            throw std::out_of_range(
                "ExpertOverlay epoch arena request slot is outside capacity");
        }
        return static_cast<std::size_t>(slot);
    }

    DeviceMoEOverlayEpochTicket *DeviceMoEOverlayEpochArena::requestTicket(
        std::uint32_t slot)
    {
        return request_tickets_ + checkedSlot(slot);
    }

    DeviceMoEOverlayEpochStatus *DeviceMoEOverlayEpochArena::requestStatus(
        std::uint32_t slot)
    {
        return request_statuses_ + checkedSlot(slot);
    }

    std::size_t DeviceMoEOverlayEpochArena::allocationBytes() const noexcept
    {
        return sizeof(DeviceMoEOverlayEpochControl) +
               static_cast<std::size_t>(request_slot_capacity_) *
                   sizeof(DeviceMoEOverlayEpochTicket) +
               static_cast<std::size_t>(request_slot_capacity_) *
                   sizeof(DeviceMoEOverlayEpochStatus) +
               sizeof(std::uint64_t) +
               sizeof(DeviceMoEOverlayEpochStatus);
    }

    void DeviceMoEOverlayEpochArena::release() noexcept
    {
        if (device_id_.is_cpu())
        {
            delete maintenance_status_;
            delete maintenance_epoch_;
            delete[] request_statuses_;
            delete[] request_tickets_;
            delete control_;
        }
        else if (backend_)
        {
            const int ordinal = device_id_.ordinal;
            const auto release_one = [&](void *pointer, const char *label)
            {
                if (!pointer)
                    return;
                try
                {
                    backend_->free(pointer, ordinal);
                }
                catch (const std::exception &error)
                {
                    LOG_ERROR("[DeviceMoEOverlayEpochArena] Failed to free "
                              << label << " on " << device_id_.to_string()
                              << ": " << error.what());
                }
                catch (...)
                {
                    LOG_ERROR("[DeviceMoEOverlayEpochArena] Failed to free "
                              << label << " on " << device_id_.to_string());
                }
            };
            release_one(maintenance_status_, "maintenance status");
            release_one(maintenance_epoch_, "maintenance epoch");
            release_one(request_statuses_, "request statuses");
            release_one(request_tickets_, "request tickets");
            release_one(control_, "control");
        }

        maintenance_status_ = nullptr;
        maintenance_epoch_ = nullptr;
        request_statuses_ = nullptr;
        request_tickets_ = nullptr;
        control_ = nullptr;
        backend_ = nullptr;
    }
} // namespace llaminar2
