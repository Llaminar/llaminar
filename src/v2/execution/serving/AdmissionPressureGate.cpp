/**
 * @file AdmissionPressureGate.cpp
 * @brief Pressure-gate decision implementation.
 */
#include "AdmissionPressureGate.h"

namespace llaminar2
{
PressureAdmissionResult decidePressureAdmission(
    const AllResourceWitnessResult &current,
    const AllResourceWitnessResult &proposed,
    std::size_t queued_depth, std::size_t queue_bound_q)
{
    // The admitted queue Q is exact-or-reject (design §4.1): beyond it the
    // request is Refuse typed, never silently accepted.
    if (queued_depth >= queue_bound_q)
        return PressureAdmissionResult{PressureAdmission::Refuse,
            "admitted queue bound Q exhausted (backpressure refuses, not drops)"};
    // A broken current witness means the live set is already unsafe: no new
    // work may be admitted while the proof fails (fail-closed, never a
    // hopeful admission on an unproved state).
    if (!current.valid)
        return PressureAdmissionResult{PressureAdmission::Refuse,
            "current completion witness is invalid; no new work admitted"};
    // The PROPOSED witness is the only authority for admitting the request:
    // its peak (frames, banks, leases) must still have a bounded completion
    // order together with the live set. If it fails, the request STAYS QUEUED
    // (bounded backpressure; the witness is re-evaluated on the next freed
    // slot) rather than being dropped or admitted on a broken proof.
    if (!proposed.valid)
        return PressureAdmissionResult{PressureAdmission::StayQueued,
            "proposed completion witness does not hold; request stays queued"};
    return PressureAdmissionResult{PressureAdmission::Admit, {}};
}
} // namespace llaminar2
