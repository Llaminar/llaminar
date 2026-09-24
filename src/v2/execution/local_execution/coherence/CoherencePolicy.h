/**
 * @file CoherencePolicy.h
 * @brief Declarative tensor-preparation policy for compute-graph stages.
 *
 * The policy is intentionally data-only. Device movement and producer
 * publication are owned by BufferArena, DeviceGraphExecutor, and
 * TransferEngine; this header exposes no alternate coherence mechanism.
 */

#pragma once

namespace llaminar2
{
    /**
     * @brief Select which stage buffer roles the graph executor prepares.
     */
    enum class CoherencePolicy
    {
        NONE,   ///< The stage owns an explicit transfer or collective contract.
        INPUT,  ///< Prepare declared inputs; the stage owns output publication.
        OUTPUT, ///< Prepare/publish declared outputs; inputs are stage-owned.
        FULL,   ///< Prepare declared inputs and outputs.
    };

    /**
     * @brief Return a stable diagnostic name for a coherence policy.
     */
    [[nodiscard]] constexpr const char *toString(CoherencePolicy policy) noexcept
    {
        switch (policy)
        {
        case CoherencePolicy::NONE:
            return "NONE";
        case CoherencePolicy::INPUT:
            return "INPUT";
        case CoherencePolicy::OUTPUT:
            return "OUTPUT";
        case CoherencePolicy::FULL:
            return "FULL";
        }
        return "UNKNOWN";
    }
} // namespace llaminar2
