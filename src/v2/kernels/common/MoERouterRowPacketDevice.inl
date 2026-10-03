/**
 * @file MoERouterRowPacketDevice.inl
 * @brief Common CUDA/HIP lossless publication of acquired owned-router packets.
 *
 * Every live row has one source. Selection bits are copied, never reduced or
 * recomputed. Invalid extents trap before a packet is read; inactive original
 * rows receive the established -1/0 sentinels, so a smaller retained replay
 * cannot resurrect routes from a previous larger request.
 */
#pragma once
#include "MoERouterRowPacket.h"

namespace
{
    /** @brief Reject corrupt acquired metadata before accessing packet payload. */
    __device__ __forceinline__ void requireRouterPacket(bool valid)
    {
        if (!valid)
        {
#if defined(__CUDA_ARCH__)
            __trap();
#else
            __builtin_trap();
#endif
        }
    }

    /** @brief Parallel direct publication; all storage was bound before capture.
     * @param p Canonical device banks/counts and immutable domain geometry.
     * @param stride Uniform physical peer slot size, never the live wire count. */
    __global__ void publish_router_owned_rows_kernel(
        llaminar2::MoERouterRowPublication p, std::size_t stride)
    {
        using llaminar2::DeviceRowPartition;
        using llaminar2::MoERouterSelectedRoute;
        const auto &layout = p.layout;
        const auto &partition = layout.partition;
        const int live = *p.live_rows;
        requireRouterPacket(partition.resolveFor(layout.capacity, live).valid());
        const std::size_t index = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
        // Authenticate even sources with no rows. Every active payload read
        // below also authenticates its own source, so concurrent blocks cannot
        // race ahead of a failing metadata check and read an incomplete packet.
        if (index < std::size_t(partition.participants()))
        {
            const auto span = DeviceRowPartition::resolveMember(
                layout.capacity, live, int(index), partition.participants());
            requireRouterPacket(p.acquired_bytes[index] ==
                std::uint64_t(span.count) * layout.top_k * sizeof(MoERouterSelectedRoute));
        }
        if (index >= std::size_t(layout.capacity) * layout.top_k) return;
        const int row = int(index / layout.top_k);
        if (row >= live)
        {
            p.indices[index] = -1.0f;
            p.weights[index] = 0.0f;
            return;
        }
        const int source = partition.ownerFor(layout.capacity, live, row);
        const auto span = DeviceRowPartition::resolveMember(
            layout.capacity, live, source, partition.participants());
        requireRouterPacket(source >= 0 && span.valid() && p.acquired_bytes[source] ==
            std::uint64_t(span.count) * layout.top_k * sizeof(MoERouterSelectedRoute));
        const auto *packet = p.local;
        if (source != partition.participant())
        {
            const int slot = source < partition.participant() ? source : source - 1;
            packet = reinterpret_cast<const MoERouterSelectedRoute *>(
                static_cast<const unsigned char *>(p.peers) + std::size_t(slot) * stride);
        }
        const auto selected = packet[index - std::size_t(span.first) * layout.top_k];
        p.indices[index] = selected.expert;
        p.weights[index] = selected.weight;
    }
}
