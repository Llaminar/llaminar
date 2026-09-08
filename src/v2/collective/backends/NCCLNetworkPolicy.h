/**
 * @file NCCLNetworkPolicy.h
 * @brief Declarative network-module policy for NCCL communicators.
 *
 * NCCL distinguishes CUDA-local transports, such as peer access and shared
 * memory, from the network module used when those transports cannot carry a
 * connection. A communicator confined to one node must never probe an RDMA
 * provider merely because one happens to be installed on the host. Apart from
 * adding startup work, that probe can enter provider code which is unrelated
 * to the communicator's declared topology.
 *
 * This header keeps the topology-to-network decision independent of NCCL's C
 * ABI so it can be proved by ordinary CPU unit tests. The selected module is
 * passed to ncclCommInitRankConfig for each communicator; no process-global
 * environment variable is mutated.
 */

#pragma once

#include "../DeviceGroup.h"

#include <string_view>

namespace llaminar2
{

    /**
     * @brief Network module requested for a newly constructed NCCL communicator.
     *
     * Automatic preserves NCCL's normal inter-node transport discovery.
     * Socket constrains only the network fallback inside a node-local
     * communicator; CUDA P2P and NCCL shared-memory transports remain enabled.
     */
    enum class NCCLNetworkModule
    {
        Automatic,
        Socket,
    };

    /**
     * @brief Select the NCCL network module from the declared collective scope.
     *
     * LOCAL is a same-node contract, so its network fallback is always Socket.
     * GLOBAL and HYBRID may cross nodes and therefore retain NCCL's automatic
     * plugin selection, including InfiniBand or RoCE where available.
     *
     * @param scope Declared scope of the communicator's DeviceGroup.
     * @return The network module required by that topology.
     */
    [[nodiscard]] constexpr NCCLNetworkModule selectNCCLNetworkModule(
        CollectiveScope scope) noexcept
    {
        return scope == CollectiveScope::LOCAL
                   ? NCCLNetworkModule::Socket
                   : NCCLNetworkModule::Automatic;
    }

    /**
     * @brief Convert a typed policy to the NCCL module name.
     *
     * An empty name means that ncclCommInitRank should retain automatic network
     * selection. Non-empty names are supplied through ncclConfig_t::netName.
     *
     * @param module Typed network policy.
     * @return "Socket" for a node-local communicator, otherwise an empty view.
     */
    [[nodiscard]] constexpr std::string_view ncclNetworkModuleName(
        NCCLNetworkModule module) noexcept
    {
        return module == NCCLNetworkModule::Socket
                   ? std::string_view{"Socket"}
                   : std::string_view{};
    }

} // namespace llaminar2
