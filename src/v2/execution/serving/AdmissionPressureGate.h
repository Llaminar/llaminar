/**
 * @file AdmissionPressureGate.h
 * @brief Typed pool-pressure admission: witness-proofed acceptance of new work.
 *
 * Plan §5.1 pressure/progress: before a queued request may consume a freed
 * pool slot, the ALL-RESOURCE completion witness must still hold with the new
 * work added. This gate composes the completion witness (frames, table banks,
 * effective-use leases) with the serving admission outcome: a request is
 * admitted only when its admission leaves a finite completion order for every
 * class — otherwise it stays queued (or is refused when the queue is full).
 *
 * It is a PURE policy input: no live ledger, no physical accounting, PMA
 * remains the sole authority. The caller supplies the admitted maxima and the
 * witness result; this gate returns the typed decision.
 */
#pragma once
#include "execution/serving/ServingBatch.h"
#include "planning/AllResourceCompletionWitness.h"
#include <cstddef>
#include <string>

namespace llaminar2
{
/** @brief Typed pressure-gate decision (no silent drop, no implicit queue). */
enum class PressureAdmission : std::uint8_t
{
    Admit,   ///< Completion witness holds with the new request: release it.
    StayQueued, ///< Would violate the witness: keep it queued (bounded backpressure).
    Refuse,  ///< Queue is full or request invalid: refuse typed.
};

/** @brief Admission pressure gate result with a precise diagnostic. */
struct PressureAdmissionResult
{
    PressureAdmission admission = PressureAdmission::Refuse;
    std::string refusal; ///< Nonempty exactly when not Admit.
    [[nodiscard]] bool admitted() const noexcept
    {
        return admission == PressureAdmission::Admit;
    }
};

/**
 * @brief Decide one queued request under the composed completion witness.
 *
 * @param current Result of the all-resource witness for the CURRENT live set
 * (already proved at the last admission boundary).
 * @param proposed Result after adding the queued request's declared peak
 * (computed by the caller through proveAllResourceCompletion with the new
 * request included; this gate NEVER invents maxima).
 * @param queued_depth Current queue depth and the admitted queue bound Q:
 * a request beyond Q is Refuse, never silently accepted.
 * @return Admit only when the PROPOSED witness is valid; StayQueued when the
 * proposed witness fails (the queue backpressure holds it); Refuse on Q.
 */
PressureAdmissionResult decidePressureAdmission(
    const AllResourceWitnessResult &current,
    const AllResourceWitnessResult &proposed,
    std::size_t queued_depth, std::size_t queue_bound_q);
} // namespace llaminar2
