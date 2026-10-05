/**
 * @file NCCLNetworkPolicy.h
 * @brief Declarative transport and measured channel policy for NCCL communicators.
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
#include <span>
#include <stdexcept>

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

    /** @brief Immutable construction profile; native algorithms remain size-adaptive. */
    enum class NCCLCommunicatorProfile { NativeAutomatic, RTX3090SharedMemoryTP2 };

    /** @brief Authenticated startup observation of one CUDA card. */
    struct NCCLCardIdentity
    {
        std::string_view name; ///< Borrowed inventory name, used only during selection.
        int ordinal = -1; ///< Exact physical CUDA ordinal.
        int architecture_major = 0; ///< Native CUDA compute capability major.
        int architecture_minor = 0; ///< Native CUDA compute capability minor.
        int multiprocessors = 0; ///< Physical SM count.
        std::size_t memory_bytes = 0; ///< Native physical capacity, not current free bytes.
    };

    /** @brief Peer reachability is known in both directions before profile selection. */
    enum class NCCLPeerAccess { Unobserved, Available, Unavailable };

    /** @brief Per-communicator construction policy, fixed before any graph is recorded. */
    struct NCCLCommunicatorPolicy
    {
        NCCLNetworkModule network = NCCLNetworkModule::Automatic; ///< Topology-owned network module.
        NCCLCommunicatorProfile profile = NCCLCommunicatorProfile::NativeAutomatic; ///< Measured economy profile.
    };

    /**
     * @brief Select only the measured same-process pair with no peer transport.
     * @param scope Declared collective scope.
     * @param cards Complete physical membership, with distinct ordinals.
     * @param directed_peers Ordered 0-to-1 and 1-to-0 observations for a pair.
     * @return Eight-channel profile for two SM86/82-SM RTX 3090 cards on SHM;
     * otherwise the native size-adaptive policy for that topology.
     */
    [[nodiscard]] inline NCCLCommunicatorPolicy selectNCCLCommunicatorPolicy(
        CollectiveScope scope, std::span<const NCCLCardIdentity> cards,
        std::span<const NCCLPeerAccess> directed_peers)
    {
        NCCLCommunicatorPolicy result{.network = selectNCCLNetworkModule(scope)};
        if (scope != CollectiveScope::LOCAL || cards.size() != 2 ||
            directed_peers.size() != 2 || cards[0].ordinal == cards[1].ordinal)
            return result;
        for (const auto &card : cards)
            if (!card.name.ends_with("RTX 3090") || card.ordinal < 0 ||
                card.architecture_major != 8 || card.architecture_minor != 6 ||
                card.multiprocessors != 82 || card.memory_bytes < (std::size_t{23} << 30) ||
                card.memory_bytes > (std::size_t{24} << 30))
                return result;
        for (const auto access : directed_peers)
            if (access != NCCLPeerAccess::Unavailable) return result;
        result.profile = NCCLCommunicatorProfile::RTX3090SharedMemoryTP2;
        return result;
    }

    /** @brief Obtain the immutable CTA budget, or zero for native automatic selection. */
    [[nodiscard]] constexpr int ncclCommunicatorChannelBudget(NCCLCommunicatorProfile profile)
    {
        switch (profile)
        {
        case NCCLCommunicatorProfile::NativeAutomatic: return 0;
        case NCCLCommunicatorProfile::RTX3090SharedMemoryTP2: return 8;
        }
        throw std::invalid_argument("Unknown NCCL communicator profile");
    }

    /** @brief Stable policy name for setup logs and PerfStats evidence. */
    [[nodiscard]] constexpr std::string_view ncclCommunicatorProfileName(NCCLCommunicatorProfile profile)
    {
        switch (profile)
        {
        case NCCLCommunicatorProfile::NativeAutomatic: return "native_automatic";
        case NCCLCommunicatorProfile::RTX3090SharedMemoryTP2: return "rtx3090_shm_tp2";
        }
        throw std::invalid_argument("Unknown NCCL communicator profile");
    }

} // namespace llaminar2
