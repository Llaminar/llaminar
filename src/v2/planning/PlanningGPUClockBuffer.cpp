/**
 * @file PlanningGPUClockBuffer.cpp
 * @brief Original-worker fixed-clock storage, capture writes and final compact readback.
 *
 * The canonical PMA owns every physical byte. TransferEngine alone allocates
 * and copies the bank; markers write admitted stable addresses inside capture.
 * No live model state, payload mirror or independent capacity ledger is added.
 */
#include "planning/PlanningGPUClockBuffer.h"
#include "backends/IWorkerGPUContext.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/PlanningGPUClockMarker.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "transfer/TransferEngine.h"
#ifdef HAVE_ROCM
#include "backends/rocm/ROCmHardwareMetadata.h"
#endif
#include <array>
#include <limits>
#include <stdexcept>

namespace llaminar2
{
    namespace
    {
        /** @return Original device's constant-frequency timebase; missing bindings are fatal. */
        std::uint64_t frequency(DeviceId device)
        {
#ifdef HAVE_CUDA
            if (device.is_cuda()) return 1'000'000'000u;
#endif
#ifdef HAVE_ROCM
            if (device.is_rocm())
            {
                const auto khz = rocmServiceWallClockRateKHz(rocmHardwareMetadata().at(device.ordinal));
                if (khz > std::numeric_limits<std::uint64_t>::max() / 1000u)
                    throw std::overflow_error("Planning wall-clock frequency overflow");
                return khz * 1000u;
            }
#endif
            throw std::invalid_argument("Planning fixed device clock is not built for the requested backend");
        }
    }

    /** @brief Backing owners retire before the exact allocation claims that admitted them. */
    struct PlanningNativeGPUClockBuffer::Impl
    {
        IWorkerGPUContext &worker;
        DeviceId device;
        ExplicitGPUStream stream;
        size_t slots;
        std::uint64_t frequency;
        bool readback_queued = false;
        TransferEngine transfers;
        PhysicalMemoryAllocationLease host_claim, device_claim;
        std::shared_ptr<DeviceTransferBuffer> device_buffer;
        std::shared_ptr<MappedHostTransferRegion> host_buffer;
        /** @brief Claim first, then materialize only through the public transfer authority. */
        Impl(IWorkerGPUContext &worker_, DeviceId device_, ExplicitGPUStream stream_, size_t slots_,
            const std::shared_ptr<PhysicalMemoryAuthority> &memory)
            : worker(worker_), device(device_), stream(stream_), slots(slots_), frequency(llaminar2::frequency(device_)),
              host_claim(memory->claimNewAllocation(DeviceId::cpu(), PhysicalMemoryOwner::ExecutionWorkspace,
                  PlanningNativeGPUClockBuffer::hostAllocationBytes(slots_))),
              device_claim(memory->claimNewAllocation(device_, PhysicalMemoryOwner::ExecutionWorkspace,
                  PlanningNativeGPUClockBuffer::deviceAllocationBytes(slots_)))
        {
            device_buffer = transfers.allocateDeviceTransferBuffer(PlanningNativeGPUClockBuffer::deviceAllocationBytes(slots), device);
            const std::array devices{device};
            host_buffer = transfers.allocateMappedHostRegion(PlanningNativeGPUClockBuffer::deviceAllocationBytes(slots), devices);
        }
        /** @brief Reject stream substitution and off-worker native operations. */
        void requireOriginal(ExplicitGPUStream submitted) const
        {
            if (!worker.ownsCurrentThread() || submitted.get() != stream.get())
                throw std::invalid_argument("Planning clock buffer requires its original worker and stream");
        }
    };

    size_t PlanningNativeGPUClockBuffer::deviceAllocationBytes(size_t slots)
    {
        if (!slots || slots > std::numeric_limits<size_t>::max() / sizeof(std::uint64_t))
            throw std::invalid_argument("Planning clock buffer requires positive representable slot geometry");
        return slots * sizeof(std::uint64_t);
    }

    size_t PlanningNativeGPUClockBuffer::hostAllocationBytes(size_t slots)
    {
        return TransferEngine::mappedHostRegionAllocationBytes(deviceAllocationBytes(slots));
    }

    PlanningNativeGPUClockBuffer::PlanningNativeGPUClockBuffer(IWorkerGPUContext &worker, DeviceId device,
        ExplicitGPUStream stream, size_t slots, const std::shared_ptr<PhysicalMemoryAuthority> &memory)
    {
        if (!memory || !device.is_gpu() || device.ordinal < 0 || worker.deviceOrdinal() != device.ordinal ||
            !worker.ownsCurrentThread() || isGraphCaptureActive())
            throw std::invalid_argument("Planning clock buffers require admitted original-worker GPU setup");
        (void)deviceAllocationBytes(slots);
        if (GPUDeviceContextPool::instance().retainExistingContext(device).get() != &worker)
            throw std::invalid_argument("Planning clock buffer cannot substitute another backend's worker at the same ordinal");
        // Resolve marker code under this original worker's native context before
        // capture. Native module lifetime owns reuse and invalidation after reset.
        bool prepared = false;
#ifdef HAVE_CUDA
        if (device.is_cuda()) prepared = prepareCUDAPlanningClock();
#endif
#ifdef HAVE_ROCM
        if (device.is_rocm()) prepared = prepareROCmPlanningClock();
#endif
        if (!prepared) throw std::runtime_error("Planning device-clock marker preparation failed");
        impl_ = std::make_unique<Impl>(worker, device, stream, slots, memory);
    }

    PlanningNativeGPUClockBuffer::~PlanningNativeGPUClockBuffer() = default;
    size_t PlanningNativeGPUClockBuffer::slotCount() const noexcept { return impl_->slots; }
    std::uint64_t PlanningNativeGPUClockBuffer::ticksPerSecond() const noexcept { return impl_->frequency; }

    void PlanningNativeGPUClockBuffer::record(size_t slot, ExplicitGPUStream stream)
    {
        auto &owner = *impl_;
        owner.requireOriginal(stream);
        if (slot >= owner.slots || !isGraphCaptureActive())
            throw std::invalid_argument("Planning clock marker requires an admitted slot inside active capture");
        auto *destination = static_cast<std::uint64_t *>(owner.device_buffer->mutableDeviceData()) + slot;
        bool recorded = false;
#ifdef HAVE_CUDA
        if (owner.device.is_cuda()) recorded = recordCUDAPlanningClock(destination, stream.get());
#endif
#ifdef HAVE_ROCM
        if (owner.device.is_rocm()) recorded = recordROCmPlanningClock(destination, stream.get());
#endif
        if (!recorded) throw std::runtime_error("Planning captured device-clock marker submission failed");
        owner.readback_queued = false;
    }

    void PlanningNativeGPUClockBuffer::enqueueReadback(ExplicitGPUStream stream)
    {
        auto &owner = *impl_;
        owner.requireOriginal(stream);
        if (isGraphCaptureActive())
            throw std::invalid_argument("Planning timestamp readback belongs after the measured graph");
        owner.transfers.enqueueDeviceToMappedHost(*owner.device_buffer, 0, *owner.host_buffer, 0,
            deviceAllocationBytes(owner.slots), owner.device, stream.get());
        owner.readback_queued = true;
    }

    std::span<const std::uint64_t> PlanningNativeGPUClockBuffer::ticks() const
    {
        const auto &owner = *impl_;
        owner.requireOriginal(owner.stream);
        if (!owner.readback_queued || owner.worker.queryStreamExecutionState(owner.stream.get(),
                "planning final clock-bank readback") != GPUStreamExecutionState::Complete)
            throw std::logic_error("Planning timestamp bank observed before original-stream completion");
        return {static_cast<const std::uint64_t *>(owner.host_buffer->mutableHostData()), owner.slots};
    }
}
