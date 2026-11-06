/**
 * @file DeviceMoEOverlayEpochArena.cpp
 * @brief Stable allocation and setup publication for ExpertOverlay epoch RCU.
 */

#include "DeviceMoEOverlayEpochArena.h"

#include "MoEOverlayDeviceEpochProtocol.h"
#include "planning/PhysicalMemoryAuthority.h"
#include <cstring>
#include <new>
#include "../../backends/BackendManager.h"
#include "../../backends/IBackend.h"
#include "../../utils/Logger.h"
#include "../../utils/PerfStatsCollector.h"

#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace llaminar2
{
    namespace
    {
        /** @brief Single checked physical layout shared by admission and materialization. */
        struct EpochStorageLayout {
            std::size_t control, parent, tickets, statuses, maintenance_epoch, maintenance_status, parent_status, bytes;
            /** @brief Lay out every record once, including native alignment and finite-parent state. */
            static EpochStorageLayout checked(std::uint32_t slots) {
                if(!slots)throw std::invalid_argument("ExpertOverlay epoch storage requires positive request capacity");
                std::size_t frontier=0u;
                const auto append=[&](std::size_t bytes,std::size_t alignment) {
                    const auto limit=std::numeric_limits<std::size_t>::max();
                    if(frontier>limit-(alignment-1u))throw std::overflow_error("ExpertOverlay epoch storage alignment overflows");
                    const auto offset=(frontier+alignment-1u)&~(alignment-1u);
                    if(bytes>limit-offset)throw std::overflow_error("ExpertOverlay epoch storage extent overflows");
                    frontier=offset+bytes;return offset;
                };
                const auto array=[&](std::size_t size) {
                    if(slots>std::numeric_limits<std::size_t>::max()/size)
                        throw std::overflow_error("ExpertOverlay epoch request storage overflows");
                    return std::size_t(slots)*size;
                };
                EpochStorageLayout layout{};
                layout.control=append(sizeof(DeviceMoEOverlayEpochControl),alignof(DeviceMoEOverlayEpochControl));
                layout.parent=append(sizeof(DeviceMoEOverlayEpochParent),alignof(DeviceMoEOverlayEpochParent));
                layout.tickets=append(array(sizeof(DeviceMoEOverlayEpochTicket)),alignof(DeviceMoEOverlayEpochTicket));
                layout.statuses=append(array(sizeof(DeviceMoEOverlayEpochStatus)),alignof(DeviceMoEOverlayEpochStatus));
                layout.maintenance_epoch=append(sizeof(std::uint64_t),alignof(std::uint64_t));
                layout.maintenance_status=append(sizeof(DeviceMoEOverlayEpochStatus),alignof(DeviceMoEOverlayEpochStatus));
                layout.parent_status=append(sizeof(DeviceMoEOverlayEpochStatus),alignof(DeviceMoEOverlayEpochStatus));
                append(0u,alignof(DeviceMoEOverlayEpochControl));layout.bytes=frontier;return layout;
            }
        };

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
          request_slot_capacity_(config.request_slot_capacity),
          memory_authority_(std::move(config.memory_authority)),
          external_admission_epoch_(config.external_admission_epoch),
          external_admission_lifetime_(
              std::move(config.external_admission_lifetime)),
          admission_barrier_(config.admission_barrier)
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
        if ((external_admission_epoch_ == nullptr) !=
            (external_admission_lifetime_ == nullptr))
        {
            throw std::invalid_argument(
                "ExpertOverlay external admission pointer and lifetime must be bound together");
        }
        if (!admission_barrier_.empty() &&
            (!admission_barrier_.valid() || !external_admission_epoch_ ||
             !external_admission_lifetime_))
        {
            throw std::invalid_argument(
                "ExpertOverlay continuation admission barrier requires a complete external admission binding");
        }

        const auto layout=EpochStorageLayout::checked(request_slot_capacity_);
        storage_bytes_=layout.bytes;
        if(!memory_authority_ || !memory_authority_->contains(device_id_))
            throw std::invalid_argument("ExpertOverlay epoch storage requires its original admitted physical memory authority");
        allocation_lease_=std::make_unique<PhysicalMemoryAllocationLease>(
            memory_authority_->claimNewAllocation(device_id_,PhysicalMemoryOwner::ExecutionWorkspace,storage_bytes_));
        try {
            if(device_id_.is_cpu())storage_=::operator new(storage_bytes_,std::align_val_t{alignof(DeviceMoEOverlayEpochControl)});
            else {
                backend_=getBackendFor(device_id_);
                if(!backend_)throw std::runtime_error("ExpertOverlay epoch storage has no original device backend");
                storage_=backend_->allocate(storage_bytes_,device_id_.toKernelDeviceIndex());
            }
            if(!storage_)throw std::runtime_error("ExpertOverlay epoch packed allocation failed");
            auto* base=static_cast<std::byte*>(storage_);
            control_=reinterpret_cast<DeviceMoEOverlayEpochControl*>(base+layout.control);
            parent_=reinterpret_cast<DeviceMoEOverlayEpochParent*>(base+layout.parent);
            request_tickets_=reinterpret_cast<DeviceMoEOverlayEpochTicket*>(base+layout.tickets);
            request_statuses_=reinterpret_cast<DeviceMoEOverlayEpochStatus*>(base+layout.statuses);
            maintenance_epoch_=reinterpret_cast<std::uint64_t*>(base+layout.maintenance_epoch);
            maintenance_status_=reinterpret_cast<DeviceMoEOverlayEpochStatus*>(base+layout.maintenance_status);
            parent_status_=reinterpret_cast<DeviceMoEOverlayEpochStatus*>(base+layout.parent_status);
            DeviceMoEOverlayEpochControl initial_control{};
            MoEOverlayDeviceEpochProtocol::initialize(initial_control,config.initial_epoch,config.initial_bank);
            initial_control.finite_parent=parent_;
            const DeviceMoEOverlayEpochParent initial_parent{.control=control_};
            const std::uint64_t initial_maintenance_epoch=config.initial_epoch+1u;
            std::vector<std::byte> initial(storage_bytes_,std::byte{});
            std::memcpy(initial.data()+layout.control,&initial_control,sizeof(initial_control));
            std::memcpy(initial.data()+layout.parent,&initial_parent,sizeof(initial_parent));
            // Zero bytes do not encode the status record's InvalidBank sentinel.
            // Publish every typed default in the same original setup upload.
            const DeviceMoEOverlayEpochStatus initial_status{};
            for(std::uint32_t slot=0u;slot<request_slot_capacity_;++slot)
                std::memcpy(initial.data()+layout.statuses+std::size_t(slot)*sizeof(initial_status),&initial_status,sizeof(initial_status));
            std::memcpy(initial.data()+layout.maintenance_status,&initial_status,sizeof(initial_status));
            std::memcpy(initial.data()+layout.parent_status,&initial_status,sizeof(initial_status));
            std::memcpy(initial.data()+layout.maintenance_epoch,&initial_maintenance_epoch,sizeof(initial_maintenance_epoch));
            if(device_id_.is_cpu())std::memcpy(storage_,initial.data(),storage_bytes_);
            else {
                const int ordinal=device_id_.toKernelDeviceIndex();
                SetupStream setup(*backend_,ordinal);
                uploadSetupObject(*backend_,ordinal,setup.get(),storage_,initial.data(),storage_bytes_,"packed epoch records");
                if(!backend_->synchronizeStream(setup.get(),ordinal))
                    throw std::runtime_error("ExpertOverlay epoch storage setup did not complete");
            }
        } catch(...) {release();throw;}

        PerfStatsCollector::addCounter(
            "memory",
            "moe_overlay_epoch_arena_allocations",
            1.0,
            "model_setup",
            device_id_.to_string(),
            {{"bytes", std::to_string(allocationBytes())},
             {"request_slots", std::to_string(request_slot_capacity_)},
             {"initial_epoch", std::to_string(config.initial_epoch)},
             {"initial_bank", std::to_string(config.initial_bank)},
             {"external_admission",
              external_admission_epoch_ ? "true" : "false"},
             {"transaction_barrier",
              admission_barrier_.valid() ? "true" : "false"}});
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

    const DeviceMoEOverlayEpochStatus *
    DeviceMoEOverlayEpochArena::requestStatus(std::uint32_t slot) const
    {
        return request_statuses_ + checkedSlot(slot);
    }

    std::size_t DeviceMoEOverlayEpochArena::allocationBytesForSlots(std::uint32_t slots)
    {
        return EpochStorageLayout::checked(slots).bytes;
    }

    std::size_t DeviceMoEOverlayEpochArena::allocationBytes() const noexcept
    {
        return storage_bytes_;
    }

    void DeviceMoEOverlayEpochArena::release() noexcept
    {
        if(storage_) {
            if(device_id_.is_cpu())::operator delete(storage_,std::align_val_t{alignof(DeviceMoEOverlayEpochControl)});
            else if(backend_) {
                try {backend_->free(storage_,device_id_.toKernelDeviceIndex());}
                catch(...) {std::terminate();} // Never withdraw a PMA claim for bytes not physically released.
            }
        }
        storage_=nullptr;control_=nullptr;parent_=nullptr;request_tickets_=nullptr;
        request_statuses_=nullptr;maintenance_epoch_=nullptr;maintenance_status_=nullptr;parent_status_=nullptr;
        backend_=nullptr;allocation_lease_.reset();memory_authority_.reset();storage_bytes_=0u;
    }
} // namespace llaminar2
