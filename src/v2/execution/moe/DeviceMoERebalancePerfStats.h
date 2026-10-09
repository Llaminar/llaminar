/**
 * @file DeviceMoERebalancePerfStats.h
 * @brief Bounded evidence of one coherent, completed native movement observation.
 *
 * Copy, apply and byte lower bounds must come from the same device publication.
 * Folding each bound independently can pair unrelated partial observations.
 * This publisher records their joint proof in a fixed-width counter witness;
 * epochs and individual request lifetimes never become diagnostic map keys.
 */
#pragma once

#include "execution/moe/DeviceMoERebalanceController.h"
#include "utils/PerfStatsCollector.h"
#include <stdexcept>

namespace llaminar2
{
    /**
     * @brief Publish a coherent completed movement proof without retaining history.
     * @param evidence Lower bounds derived from one authenticated device snapshot.
     * @param phase Production phase owning that snapshot.
     * @param device Participant owning the evidence, never a neighboring rank.
     * @param source Stable publication boundary, not a transaction identifier.
     *
     * An incomplete observation contributes no completed proof. The ordinary
     * diagnostic counters still report its partial work. Each retained sequence
     * step contains copied, applied, completed and useful-byte lower bounds from
     * the same snapshot; exact minima prove every recorded step was complete.
     */
    inline void recordCompletedDeviceMoEMovement(
        const DeviceMoERebalanceRequestMovementEvidence &evidence,
        const std::string &phase, const std::string &device, const char *source)
    {
        if (evidence.completed_payload_lower_bound == 0 ||
            evidence.useful_payload_bytes_lower_bound == 0)
            return;
        if (evidence.completed_payload_lower_bound > evidence.copied_payload_lower_bound ||
            evidence.completed_payload_lower_bound > evidence.applied_payload_lower_bound)
            throw std::logic_error("Completed MoE movement exceeds its copy/apply proof");
        PerfStatsCollector::addCounterWithSequence(
            "moe_rebalance", "device_rebalance_completed_movement_observations", 1.0,
            {evidence.copied_payload_lower_bound, evidence.applied_payload_lower_bound,
             evidence.completed_payload_lower_bound, evidence.useful_payload_bytes_lower_bound},
            phase, device, {{"source", source}});
    }
}
