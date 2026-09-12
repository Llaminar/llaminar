/**
 * @file NativeMoEMovementRequestIdentity.h
 * @brief Authenticate terminal evidence against the immutable GPU request binding.
 *
 * Both backends initialize the same request/arena/participant identity. HIP
 * additionally publishes scheduler actions; a CUDA conditional parent needs
 * no host dispatch ticket publication. Terminal movement observation consumes
 * only the shared immutable binding, never the optional action or cadence.
 */
#pragma once
#include "DeviceMoERebalanceABI.h"
#include <stdexcept>

namespace llaminar2
{
    /**
     * @brief Resolve the authenticated request epoch for a terminal archive.
     * @param ticket Device-initialized identity copied at the terminal boundary.
     * @param generation Exact captured workspace generation.
     * @param participant Configured native publication participant.
     * @param participants Immutable native domain cardinality.
     * @return Positive request epoch; archive observation checks its monotonicity.
     * @throws std::invalid_argument on missing ABI or contradictory ownership.
     *
     * This is not scheduler admission: callers submitting a named transaction
     * must still use matchesLifecycle(), including its action/health contract.
     */
    [[nodiscard]] inline std::uint64_t nativeMoEMovementArchiveRequestEpoch(
        const DeviceMoERebalanceDispatchTicket &ticket, std::uint64_t generation,
        std::uint32_t participant, std::uint32_t participants)
    {
        if (!ticket.hasValidABI() || !ticket.sessionEpoch() || !generation || !participants ||
            participant >= participants || ticket.workspaceGeneration() != generation ||
            ticket.participant_id != participant || ticket.participant_count != participants)
            throw std::invalid_argument("Native MoE movement archive has an invalid request/arena/participant binding");
        return ticket.sessionEpoch();
    }
}
