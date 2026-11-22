/**
 * @file AdmissionKindGate.h
 * @brief Request admission/lifecycle gate (design §3.1, §6 admission-kind row).
 *
 * The executor admits a request exactly once as NewRequest or ContinueResponse
 * against a typed gate. A gate that returns Refused (backpressure) must not
 * mutate request state; admission may be retried later but each admitted
 * request has exactly one admission outcome. The gate is not a wall-clock
 * policy: it reflects capacity/lease state only.
 */

#pragma once

#include "execution/config/ServingCapacity.h"
#include "execution/serving/ServingRowState.h"

#include <cstdint>
#include <stdexcept>

namespace llaminar2
{
    /** @brief Admission outcome for one request attempt. */
    enum class AdmissionOutcome : std::uint8_t
    {
        AcceptedNewRequest,
        AcceptedContinuation,
        RefusedBackpressure,
    };

    /** @brief Canonical admission gate: one outcome per admitted request, type-safe. */
    struct AdmissionGate
    {
        std::uint64_t request_id = 0;
        AdmissionOutcome outcome = AdmissionOutcome::RefusedBackpressure;
        bool settled = false;

        /** @brief Accept a NewRequest admiss ion (exactly once). */
        void admitNew(std::uint32_t slot, std::uint64_t generation)
        {
            ensureUnsettled();
            // A prior refusal is retryable: admit after refuse is legal.
            outcome = AdmissionOutcome::AcceptedNewRequest;
            settled = true;
            (void)slot;
            (void)generation;
        }

        /** @brief Accept a continuation (requires a NewRequest predecessor). */
        void admitContinuation(std::uint32_t slot, std::uint64_t generation,
                               const ServingRowState &predecessor)
        {
            ensureUnsettled();
            if (predecessor.kind != AdmissionKind::NewRequest)
                throw std::logic_error("Continuation requires a NewRequest predecessor");
            if (predecessor.slot != slot || predecessor.generation != generation)
                throw std::logic_error("Continuation must keep exact slot/generation identity");
            outcome = AdmissionOutcome::AcceptedContinuation;
            settled = true;
        }

        /** @brief Refuse with backpressure (no state mutation; RETRYABLE). */
        void refuse()
        {
            if (settled)
                throw std::logic_error("Admission already settled; cannot refuse");
            // A refusal does NOT settle: the same request may be retried later
            // (backpressure is transient). Only an admitted outcome settles.
            outcome = AdmissionOutcome::RefusedBackpressure;
        }

    private:
        void ensureUnsettled() const
        {
            if (settled)
                throw std::logic_error("Admission already settled (exactly once)");
        }
    };
} // namespace llaminar2
