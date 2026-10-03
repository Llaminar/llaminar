/**
 * @file CapturedAllGatherStage.h
 * @brief Transport-independent input-fork contract for captured byte exchanges.
 *
 * The concrete stage owns membership, packet geometry and transport. The overlap
 * builder owns the paired event DAG and publishes the receive bank only after
 * joining its exact completion event. This interface does not introduce another
 * stream, allocation, coherence ledger or live collective authority.
 */
#pragma once

#include "../IComputeStage.h"
#include "memory/BufferId.h"

namespace llaminar2
{
    class AcquiredDeviceTransferInput;

    /** @brief Immutable arena identities kept live across an exchange fork/join. */
    struct CapturedAllGatherBuffers
    {
        ITensor *input; ///< Canonical local packet owner, read-only during exchange.
        ITensor *output; ///< Disjoint canonical receive owner, published at the join.
        BufferId input_id; ///< Arena lifetime begins before the producer fork.
        BufferId output_id; ///< Arena lifetime extends through downstream consumers.
    };

    /**
     * @brief One lossless captured exchange, independent of its native transport.
     *
     * Ordinary execution publishes on its own graph stream. Forked execution
     * accepts only TransferEngine's acquired producer proof and deliberately
     * leaves publication to the enclosing paired join, never the auxiliary lane.
     */
    class CapturedAllGatherStage : public IComputeStage
    {
    public:
        /** @brief Retain the exact participant without discovering or preparing hardware. */
        explicit CapturedAllGatherStage(DeviceId device) : IComputeStage(device) {}
        /** @return Canonical banks and arena roles authenticated by the concrete stage. */
        [[nodiscard]] virtual CapturedAllGatherBuffers exchangeBuffers() const = 0;
        /**
         * @brief Validate membership/storage on the canonical producer frontier.
         * @param execution Exact main-stream authority before recording the fork.
         * @throws std::exception For unprepared, stale or foreign storage/topology.
         */
        virtual void validateEnqueue(const StageGPUExecution &execution) const = 0;
        /**
         * @brief Enqueue on the acquired auxiliary stream without publishing output.
         * @param input Unforgeable proof of the exact input owner and event wait.
         * @return Whether every transport submission succeeded.
         * @throws std::exception For an unrelated fork or invalid retained binding.
         * @note Only a paired graph join may publish the destination afterwards.
         */
        virtual bool enqueueAcquiredInput(const AcquiredDeviceTransferInput &input) const = 0;
    };
}
