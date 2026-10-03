/**
 * @file RCCLHostTransportStorage.h
 * @brief Passive native-RCCL connection storage evidence, never an allocation ledger.
 *
 * The native dependency owns its transport descriptors. This fixed-size receipt
 * reads those live descriptors at a quiescent setup/result boundary; it neither
 * reserves physical memory nor creates a second placement or byte authority.
 * Same-process pinned ownership and cross-process registered file mappings are
 * deliberately distinct. A raw process pointer is not an IPC handle.
 */
#pragma once

#include <cstdint>
#include <type_traits>

namespace llaminar2
{
    /**
     * @brief Versioned C-compatible observation of one communicator's SHM endpoints.
     *
     * Byte fields count mapped endpoint extents, not unique physical allocations.
     * Send/receive endpoints may observe the same backing; admission must never
     * consume these counters. Zero connections can legitimately mean native P2P.
     */
    struct RCCLHostTransportStorage final
    {
        std::uint64_t abi_version = 1;
        std::uint64_t record_bytes = 64;
        std::uint64_t same_process_pinned_connections = 0;
        std::uint64_t shared_process_registered_connections = 0;
        std::uint64_t cu_mem_connections = 0;
        std::uint64_t same_process_pinned_mapped_bytes = 0;
        std::uint64_t shared_process_registered_mapped_bytes = 0;
        std::uint64_t cu_mem_mapped_bytes = 0;

        /** @return Whether the dependency returned this complete, exact ABI. */
        [[nodiscard]] bool valid() const noexcept
        {
            return abi_version == 1 && record_bytes == sizeof(RCCLHostTransportStorage);
        }
    };

    static_assert(sizeof(RCCLHostTransportStorage) == 64);
    static_assert(std::is_standard_layout_v<RCCLHostTransportStorage>);
    static_assert(std::is_trivially_copyable_v<RCCLHostTransportStorage>);
}
