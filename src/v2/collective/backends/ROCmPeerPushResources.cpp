/**
 * @file ROCmPeerPushResources.cpp
 * @brief Exact uncached inbox allocation under the original collective's PMA reservation.
 *
 * Peers borrow immutable addresses; the original coordinator retains this owner
 * through all captured readers and native communicator retirement. A failed
 * initialization never publishes a usable binding and still frees physical
 * roots before returning their canonical claims. Cold synchronization here
 * completes initialization only; enqueue/replay has no host synchronization.
 */
#include "ROCmPeerPushResources.h"
#include "backends/rocm/HipDeviceGuard.h"
#include <hip/hip_runtime.h>
#include <stdexcept>
#include <string>

namespace llaminar2
{
    namespace
    {
        /** @brief Fail creation/publication immediately without changing allocation policy. */
        void requireHIP(hipError_t result, const char *operation)
        {
            if (result != hipSuccess)
                throw std::runtime_error(std::string("Peer push ") + operation + ": " + hipGetErrorString(result));
        }

        /** @brief Authenticate the original selected device and the quiescent explicit stream. */
        void requireColdOwner(DeviceId device, void *stream)
        {
            if (!device.is_rocm() || reinterpret_cast<std::uintptr_t>(stream) <= 2)
                throw std::invalid_argument("Peer push requires an original ROCm device and explicit stream");
            int current = -1;
            requireHIP(hipGetDevice(&current), "owner query");
            if (current != device.ordinal) throw std::logic_error("Peer push materialized on a foreign device");
            hipStreamCaptureStatus status;
            requireHIP(hipStreamIsCapturing(static_cast<hipStream_t>(stream), &status), "capture query");
            if (status != hipStreamCaptureStatusNone) throw std::logic_error("Peer push storage cannot grow during capture");
        }
    }

    ROCmPeerPushResources::ROCmPeerPushResources(DeviceId device, PeerPushLayout layout, int rank,
        const volatile std::uint32_t *abort, void *stream)
        : device_(device), stream_(stream), binding_{nullptr, layout, abort, rank}
    {}

    std::unique_ptr<ROCmPeerPushResources> ROCmPeerPushResources::create(DeviceId device,
        const PhysicalMemoryOwnerReservation &reserve, PeerPushLayout layout, int rank,
        const volatile std::uint32_t *abort, void *stream)
    {
        requireColdOwner(device, stream);
        if (!reserve.valid() || reserve.allocatorIdentity().device != device ||
            reserve.owner() != PhysicalMemoryOwner::LocalCollective ||
            rank < 0 || rank >= layout.participants || !abort)
            throw std::invalid_argument("Peer push requires its original admitted collective owner");
        std::size_t residency = 0;
        if (!rocmPeerPushResidentCapacity(residency)) throw std::runtime_error("Peer push occupancy query failed");
        const auto checked = PeerPushLayout::create(layout.bank_elements, layout.participants, residency);
        if (checked.bank_elements != layout.bank_elements || checked.blocks != layout.blocks)
            throw std::invalid_argument("Peer push layout changed after quotation");
        auto owner = std::unique_ptr<ROCmPeerPushResources>(new ROCmPeerPushResources(device, layout, rank, abort, stream));
        owner->slab_lease_ = reserve.claimAllocation(layout.inboxBytes() + layout.flagBytes());
        requireHIP(hipExtMallocWithFlags(&owner->slab_, owner->slab_lease_.bytes(), hipDeviceMallocUncached), "uncached inbox allocation");
        owner->table_lease_ = reserve.claimAllocation(layout.endpointBytes());
        requireHIP(hipMalloc(&owner->table_, layout.endpointBytes()), "endpoint table allocation");
        owner->endpoint_ = {static_cast<float *>(owner->slab_), reinterpret_cast<std::uint64_t *>(
            static_cast<unsigned char *>(owner->slab_) + layout.inboxBytes())};
        owner->binding_.endpoints = owner->table_;
        requireHIP(hipMemsetAsync(owner->endpoint_.flags, 0, layout.flagBytes(), static_cast<hipStream_t>(stream)), "initial flag publication");
        owner->state_ = State::Unpublished;
        return owner;
    }

    ROCmPeerPushResources::~ROCmPeerPushResources()
    {
        if (!slab_ && !table_) return;
        int previous = -1;
        if (hipGetDevice(&previous) != hipSuccess || HipDeviceGuard::forceSetDevice(device_.ordinal) != 0)
            std::terminate();
        // A failed cold publication can still own queued memset/copy work.
        // Ready owners arrive only after the coordinator retires all readers.
        if (state_ != State::Ready && hipStreamSynchronize(static_cast<hipStream_t>(stream_)) != hipSuccess)
            std::terminate();
        if (table_ && hipFree(table_) != hipSuccess) std::terminate();
        if (slab_ && hipFree(slab_) != hipSuccess) std::terminate();
        if (HipDeviceGuard::forceSetDevice(previous) != 0) std::terminate();
    }

    void ROCmPeerPushResources::publish(std::span<const PeerPushEndpoint> peers, void *stream)
    {
        requireColdOwner(device_, stream);
        if (state_ != State::Unpublished || stream != stream_ || peers.size() != std::size_t(binding_.layout.participants) ||
            peers[binding_.rank].inbox != endpoint_.inbox || peers[binding_.rank].flags != endpoint_.flags)
            throw std::logic_error("Peer push publication changed its original lifetime or membership");
        for (std::size_t i = 0; i < peers.size(); ++i)
        {
            if (!peers[i].inbox || !peers[i].flags || reinterpret_cast<std::uintptr_t>(peers[i].inbox) % 16 ||
                reinterpret_cast<std::uintptr_t>(peers[i].flags) % 64)
                throw std::invalid_argument("Peer push has an absent or unaligned original endpoint");
            for (std::size_t j = 0; j < i; ++j)
                if (peers[i].inbox == peers[j].inbox || peers[i].flags == peers[j].flags)
                    throw std::invalid_argument("Peer push contains an aliased participant");
        }
        state_ = State::Publishing;
        requireHIP(hipMemcpyAsync(table_, peers.data(), binding_.layout.endpointBytes(), hipMemcpyHostToDevice,
                                 static_cast<hipStream_t>(stream)), "peer directory publication");
        requireHIP(hipStreamSynchronize(static_cast<hipStream_t>(stream)), "cold publication completion");
        state_ = State::Ready;
    }

    const PeerPushBinding &ROCmPeerPushResources::binding() const
    {
        if (state_ != State::Ready) throw std::logic_error("Unpublished peer push storage cannot execute");
        return binding_;
    }
}
