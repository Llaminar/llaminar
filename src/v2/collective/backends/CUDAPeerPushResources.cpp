/**
 * @file CUDAPeerPushResources.cpp
 * @brief Exact PMA-owned CUDA P2P inbox and endpoint-table materialization.
 */
#include "CUDAPeerPushResources.h"
#include "utils/Logger.h"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace llaminar2
{
    namespace
    {
        void requireCUDA(cudaError_t result, const char *operation)
        {
            if (result != cudaSuccess)
                throw std::runtime_error(std::string("CUDA peer push ") + operation + ": " + cudaGetErrorString(result));
        }

        void requireColdOwner(DeviceId device, void *stream)
        {
            if (!device.is_cuda() || reinterpret_cast<std::uintptr_t>(stream) <= 2u)
                throw std::invalid_argument("CUDA peer push requires an original CUDA device and explicit stream");
            int current = -1;
            requireCUDA(cudaGetDevice(&current), "owner query");
            if (current != device.ordinal)
                throw std::logic_error("CUDA peer push materialized on a foreign device");
            cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
            requireCUDA(cudaStreamIsCapturing(static_cast<cudaStream_t>(stream), &status), "capture query");
            if (status != cudaStreamCaptureStatusNone)
                throw std::logic_error("CUDA peer push storage cannot grow during capture");
        }
    } // namespace

    CUDAPeerPushResources::CUDAPeerPushResources(DeviceId device, PeerPushLayout layout, int rank,
        const volatile std::uint32_t *abort, void *stream)
        : device_(device), stream_(stream), binding_{nullptr, layout, abort, rank}
    {}

    std::unique_ptr<CUDAPeerPushResources> CUDAPeerPushResources::create(DeviceId device,
        const PhysicalMemoryOwnerReservation &reserve, PeerPushLayout layout, int rank,
        const volatile std::uint32_t *abort, void *stream)
    {
        requireColdOwner(device, stream);
        if (!reserve.valid() || reserve.allocatorIdentity().device != device ||
            reserve.owner() != PhysicalMemoryOwner::LocalCollective ||
            rank < 0 || rank >= layout.participants || !abort)
            throw std::invalid_argument("CUDA peer push requires its original admitted collective owner");
        std::size_t residency = 0;
        if (!cudaPeerPushResidentCapacity(residency))
            throw std::runtime_error("CUDA peer push occupancy query failed");
        const auto checked = PeerPushLayout::create(layout.bank_elements, layout.participants, residency);
        if (checked.bank_elements != layout.bank_elements || checked.blocks != layout.blocks)
            throw std::invalid_argument("CUDA peer push layout changed after quotation");

        auto owner = std::unique_ptr<CUDAPeerPushResources>(
            new CUDAPeerPushResources(device, layout, rank, abort, stream));
        owner->slab_lease_ = reserve.claimAllocation(layout.inboxBytes() + layout.flagBytes());
        requireCUDA(cudaMalloc(&owner->slab_, owner->slab_lease_.bytes()), "inbox allocation");
        owner->table_lease_ = reserve.claimAllocation(layout.endpointBytes());
        requireCUDA(cudaMalloc(&owner->table_, owner->table_lease_.bytes()), "endpoint table allocation");
        owner->endpoint_ = {
            static_cast<float *>(owner->slab_),
            reinterpret_cast<std::uint64_t *>(
                static_cast<unsigned char *>(owner->slab_) + layout.inboxBytes()),
        };
        owner->binding_.endpoints = owner->table_;
        requireCUDA(cudaMemsetAsync(owner->endpoint_.flags, 0, layout.flagBytes(),
            static_cast<cudaStream_t>(stream)), "initial flag zero");
        owner->state_ = State::Unpublished;
        return owner;
    }

    CUDAPeerPushResources::~CUDAPeerPushResources()
    {
        if (!slab_ && !table_) return;
        int previous = -1;
        const bool had_prev = (cudaGetDevice(&previous) == cudaSuccess);
        if (cudaSetDevice(device_.ordinal) != cudaSuccess)
            std::terminate();
        if (state_ != State::Ready && cudaStreamSynchronize(static_cast<cudaStream_t>(stream_)) != cudaSuccess)
            std::terminate();
        if (table_ && cudaFree(table_) != cudaSuccess)
            std::terminate();
        if (slab_ && cudaFree(slab_) != cudaSuccess)
            std::terminate();
        if (had_prev && previous >= 0 && previous != device_.ordinal)
            (void)cudaSetDevice(previous);
    }

    void CUDAPeerPushResources::publish(std::span<const PeerPushEndpoint> peers, void *stream)
    {
        requireColdOwner(device_, stream);
        if (state_ != State::Unpublished || stream != stream_ ||
            peers.size() != std::size_t(binding_.layout.participants) ||
            peers[binding_.rank].inbox != endpoint_.inbox ||
            peers[binding_.rank].flags != endpoint_.flags)
            throw std::logic_error("CUDA peer push publication changed its original lifetime or membership");
        for (std::size_t i = 0; i < peers.size(); ++i)
        {
            if (!peers[i].inbox || !peers[i].flags ||
                reinterpret_cast<std::uintptr_t>(peers[i].inbox) % 16u != 0u ||
                reinterpret_cast<std::uintptr_t>(peers[i].flags) % 64u != 0u)
                throw std::invalid_argument("CUDA peer push has an absent or unaligned original endpoint");
            for (std::size_t j = 0; j < i; ++j)
                if (peers[i].inbox == peers[j].inbox || peers[i].flags == peers[j].flags)
                    throw std::invalid_argument("CUDA peer push contains an aliased participant");
        }
        state_ = State::Publishing;
        requireCUDA(cudaMemcpyAsync(table_, peers.data(), binding_.layout.endpointBytes(),
            cudaMemcpyHostToDevice, static_cast<cudaStream_t>(stream)), "peer directory publication");
        if (const auto result = cudaStreamSynchronize(static_cast<cudaStream_t>(stream)); result != cudaSuccess)
        {
            LOG_ERROR("CUDA peer push cold publication completion failed: " << cudaGetErrorString(result));
            std::terminate();
        }
        state_ = State::Ready;
    }

    const PeerPushBinding &CUDAPeerPushResources::binding() const
    {
        if (state_ != State::Ready)
            throw std::logic_error("Unpublished CUDA peer push storage cannot execute");
        return binding_;
    }
} // namespace llaminar2
