#pragma once

/**
 * @file MTPVerifierPolicy.h
 * @brief Production policy for choosing an MTP verifier execution lane.
 *
 * The policy is intentionally fail-closed.  A production MTP verifier may use
 * direct all-position state publication when that stronger continuation proof
 * exists, or it uses grouped decode-equivalent verifier outcomes as the
 * mandatory baseline.  Serial row replay remains a diagnostic/oracle concept
 * owned by focused tests and helper utilities; it is not a production fallback
 * selected from this policy.
 */

namespace llaminar2
{

    enum class MTPVerifierExecutionPath
    {
        Unsupported,
        AllPositionStatePublication,
        GroupedDecodeEquivalentOutcome,
    };

    /**
     * @brief Inputs that choose the verifier execution contract for one MTP step.
     *
     * Direct all-position publication is a stronger path and therefore remains
     * an explicit input.  Grouped decode-equivalent verification is not a
     * capability advertisement anymore: every production backend is expected to
     * implement it.  If the selected runner later lacks the structural
     * publisher required for the grouped outcome, the caller must fail loudly
     * instead of falling back to serial row replay.
     */
    struct MTPVerifierPolicyInput
    {
        bool greedy_sampling = false;
        bool stochastic_verify = false;
        bool uses_sampling_penalties = false;
        bool supports_row_local_penalty_application = false;
        bool supports_spec_state_publication = false;
    };

    struct MTPVerifierPolicyDecision
    {
        MTPVerifierExecutionPath path =
            MTPVerifierExecutionPath::Unsupported;
        const char *reason = "decode_equivalent_verifier_unavailable";
    };

    inline MTPVerifierPolicyDecision chooseMTPVerifierPolicy(
        const MTPVerifierPolicyInput &input)
    {
        MTPVerifierPolicyDecision decision;

        const bool supported_sampling_mode =
            input.greedy_sampling || input.stochastic_verify;
        const bool row_local_penalties_supported =
            !input.uses_sampling_penalties ||
            input.supports_row_local_penalty_application;

        if (!supported_sampling_mode)
        {
            decision.reason =
                "sampling_mode_not_supported_by_grouped_verifier";
            return decision;
        }

        if (!row_local_penalties_supported)
        {
            decision.reason =
                "row_local_penalty_application_required_for_grouped_verifier";
            return decision;
        }

        if (supported_sampling_mode &&
            input.supports_spec_state_publication)
        {
            decision.path = MTPVerifierExecutionPath::AllPositionStatePublication;
            decision.reason =
                input.uses_sampling_penalties
                    ? "greedy_penalties_use_all_position_state_publication"
                    : (input.stochastic_verify
                           ? "stochastic_uses_all_position_state_publication"
                           : "greedy_uses_all_position_state_publication");
            return decision;
        }

        /*
         * Grouped verifier publication is now the target-state baseline.  The
         * policy no longer asks a runner to advertise this as an optional
         * feature because doing so made production correctness depend on stale
         * capability tables.  Route selection below this policy chooses the
         * concrete structural publisher, and missing grouped infrastructure
         * fails at that boundary.
         */
        decision.path =
            MTPVerifierExecutionPath::GroupedDecodeEquivalentOutcome;
        decision.reason =
            input.uses_sampling_penalties
                ? "greedy_penalties_use_grouped_decode_equivalent_outcome"
                : (input.stochastic_verify
                       ? "stochastic_uses_grouped_decode_equivalent_outcome"
                       : "greedy_uses_grouped_decode_equivalent_outcome");
        return decision;
    }

} // namespace llaminar2
