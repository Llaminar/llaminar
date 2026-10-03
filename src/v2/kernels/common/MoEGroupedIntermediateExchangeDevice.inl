/**
 * @file MoEGroupedIntermediateExchangeDevice.inl
 * @brief Shared CUDA/HIP bit-copy kernels for expert intermediate exchange.
 *
 * Each workgroup owns one original router slot at a time; no atomics or FP32
 * arithmetic occur. Explicit collective/TransferEngine stream edges protect
 * packet lifetime. Fixed native packets overwrite every capacity word; compact
 * packets overwrite only live records and publish their exact byte extent.
 * Compact consumers never read beyond that extent, even after a larger epoch.
 */
#pragma once

#include "execution/moe/MoEGroupedIntermediateExchangeABI.h"

namespace llaminar2::moe_intermediate_exchange_device
{
    // Each vendor compiles this arithmetic-free implementation into its own TU.
    // Internal linkage keeps CUDA/HIP registration stubs distinct in a binary
    // containing both backends, without maintaining two copies of the protocol.
    namespace
    {
    /** @brief Kernel arguments distilled once at capture, without device division. */
    struct Geometry
    {
        std::size_t value_words;
        std::size_t scale_words;
        std::size_t route_words;
        std::size_t participant_words;
    };

    /** @brief Fail the current device transaction rather than publish bad bytes. */
    __device__ __forceinline__ void invalidBinding()
    {
#ifdef __HIP_DEVICE_COMPILE__
        __builtin_trap();
#else
        asm volatile("trap;");
#endif
    }

    /** @brief Copy four existing bytes without aliasing float storage as integers. */
    __device__ __forceinline__ std::uint32_t readWord(const void *source, std::size_t word)
    {
        std::uint32_t value;
        __builtin_memcpy(&value, static_cast<const std::uint8_t *>(source) + word * sizeof(value), sizeof(value));
        return value;
    }

    /** @brief Publish exactly the original representation, including scale mantissas. */
    __device__ __forceinline__ void writeWord(void *destination, std::size_t word, std::uint32_t value)
    {
        __builtin_memcpy(static_cast<std::uint8_t *>(destination) + word * sizeof(value), &value, sizeof(value));
    }

    /** @brief Pack a participant's unique live routes and zero every other word. */
    __global__ __launch_bounds__(128)
    void packGroupedIntermediateKernel(MoEGroupedIntermediatePackLaunch launch, Geometry geometry)
    {
        for (std::uint32_t route = blockIdx.x; route < launch.layout.route_capacity; route += gridDim.x)
        {
            const int owner = launch.route_owners[route];
            if (owner < -1 || owner >= static_cast<int>(launch.layout.participants)) invalidBinding();
            const bool owned = owner == launch.participant;
            const int grouped = launch.original_to_grouped[route];
            if (owned ? (grouped < 0 || grouped >= static_cast<int>(launch.layout.route_capacity)) : grouped != -1)
                invalidBinding();
            for (std::size_t word = threadIdx.x; word < geometry.route_words; word += blockDim.x)
            {
                std::uint32_t value = 0;
                if (owned)
                    value = word < geometry.value_words
                        ? readWord(launch.grouped_values, grouped * geometry.value_words + word)
                        : readWord(launch.grouped_scales, grouped * geometry.scale_words + word - geometry.value_words);
                launch.packet[route * geometry.route_words + word] = value;
            }
        }
    }

    /** @brief Restore original bytes into a possibly different expert grouping. */
    __global__ __launch_bounds__(128)
    void consumeGroupedIntermediateKernel(MoEGroupedIntermediateConsumeLaunch launch, Geometry geometry)
    {
        for (std::uint32_t route = blockIdx.x; route < launch.layout.route_capacity; route += gridDim.x)
        {
            const int owner = launch.route_owners[route];
            const int grouped = launch.original_to_grouped[route];
            if (owner < -1 || owner >= static_cast<int>(launch.layout.participants)) invalidBinding();
            if (owner == -1)
            {
                if (grouped != -1) invalidBinding();
                continue;
            }
            if (grouped < 0 || grouped >= static_cast<int>(launch.layout.route_capacity)) invalidBinding();
            const auto *packet = launch.participant_packets + owner * geometry.participant_words + route * geometry.route_words;
            for (std::size_t word = threadIdx.x; word < geometry.route_words; word += blockDim.x)
            {
                const std::uint32_t value = packet[word];
                if (word < geometry.value_words)
                    writeWord(launch.grouped_values, grouped * geometry.value_words + word, value);
                else
                    writeWord(launch.grouped_scales, grouped * geometry.scale_words + word - geometry.value_words, value);
            }
        }
    }

    /**
     * @brief Reuse the producer's dense grouping as the compact wire order.
     * @param compact Frozen packet and authoritative last-expert grouping fields.
     * @param geometry Existing value/scale strides; no weight format conversion.
     *
     * Each original route owns a unique grouped index. Writing its ID beside
     * the existing bytes removes all per-participant holes without an atomic
     * append, extra prefix scan, or change to the floating arithmetic order.
     */
    __global__ __launch_bounds__(128)
    void packCompactIntermediateKernel(MoECompactIntermediatePackLaunch compact, Geometry geometry)
    {
        const auto &launch = compact.payload;
        const int last_offset = *compact.last_group_offset;
        const int last_count = *compact.last_group_count;
        const auto count = static_cast<std::uint64_t>(last_offset) + static_cast<std::uint64_t>(last_count);
        if (last_offset < 0 || last_count < 0 || count > launch.layout.route_capacity) invalidBinding();
        const auto record_words = geometry.route_words + 1;
        if (blockIdx.x == 0 && threadIdx.x == 0) *compact.packet_bytes = count * record_words * sizeof(std::uint32_t);
        for (std::uint32_t route = blockIdx.x; route < launch.layout.route_capacity; route += gridDim.x)
        {
            const int owner = launch.route_owners[route];
            const int grouped = launch.original_to_grouped[route];
            if (owner < -1 || owner >= static_cast<int>(launch.layout.participants)) invalidBinding();
            if (owner != launch.participant)
            {
                if (grouped != -1) invalidBinding();
                continue;
            }
            if (grouped < 0 || static_cast<std::uint64_t>(grouped) >= count) invalidBinding();
            auto *record = launch.packet + static_cast<std::size_t>(grouped) * record_words;
            if (threadIdx.x == 0) record[0] = route;
            for (std::size_t word = threadIdx.x; word < geometry.route_words; word += blockDim.x)
                record[1 + word] = word < geometry.value_words
                    ? readWord(launch.grouped_values, grouped * geometry.value_words + word)
                    : readWord(launch.grouped_scales, grouped * geometry.scale_words + word - geometry.value_words);
        }
    }

    /** @brief Scatter one acquired compact packet by original route identity.
     * @param launch Packet, acquired byte count and consumer's current grouping.
     * @param geometry Unchanged activation representation/strides.
     *
     * Owners write disjoint output rows. Records can arrive in any owner or
     * grouping order; down GEMM and the final top-k fold keep their canonical
     * arithmetic. Empty packets read no record, even after a full prior epoch.
     */
    __global__ __launch_bounds__(128)
    void consumeCompactIntermediateKernel(MoECompactIntermediateConsumeLaunch launch, Geometry geometry)
    {
        const auto bytes = *launch.packet_bytes;
        const auto record_words = geometry.route_words + 1;
        const auto record_bytes = record_words * sizeof(std::uint32_t);
        if (bytes % record_bytes != 0 || bytes / record_bytes > launch.layout.route_capacity) invalidBinding();
        const auto count = bytes / record_bytes;
        for (std::uint32_t index = blockIdx.x; index < count; index += gridDim.x)
        {
            const auto *record = launch.packet + static_cast<std::size_t>(index) * record_words;
            const auto route = record[0];
            if (route >= launch.layout.route_capacity || launch.route_owners[route] != launch.participant) invalidBinding();
            const int grouped = launch.original_to_grouped[route];
            if (grouped < 0 || grouped >= static_cast<int>(launch.layout.route_capacity)) invalidBinding();
            for (std::size_t word = threadIdx.x; word < geometry.route_words; word += blockDim.x)
            {
                const auto value = record[1 + word];
                if (word < geometry.value_words)
                    writeWord(launch.grouped_values, grouped * geometry.value_words + word, value);
                else
                    writeWord(launch.grouped_scales, grouped * geometry.scale_words + word - geometry.value_words, value);
            }
        }
    }
    } // namespace
} // namespace llaminar2::moe_intermediate_exchange_device
