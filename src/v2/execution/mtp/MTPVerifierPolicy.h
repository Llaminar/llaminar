#pragma once

namespace llaminar2
{

    enum class MTPVerifierExecutionPath
    {
        Unsupported,
        AllPositionStatePublication,
        GroupedDecodeEquivalentOutcome,
        DecodeEquivalentSequential,
    };

    /**
     * @brief Inputs that choose the verifier execution contract for one MTP step.
     *
     * The policy deliberately separates three increasingly strong contracts:
     * correctness-only serial replay, grouped verifier outcomes, and direct
     * accepted-state publication.  Phase 9.8 depends on that distinction because
     * a runner can prove M=2..4 verifier rows numerically decode-equivalent
     * before it is safe to publish live KV/GDN state from those rows.
     */
    struct MTPVerifierPolicyInput
    {
        bool greedy_sampling = false;
        bool stochastic_verify = false;
        bool uses_sampling_penalties = false;
        bool supports_row_local_penalty_application = false;
        bool supports_spec_state_publication = false;
        bool supports_grouped_decode_equivalent_outcome = false;
    };

    struct MTPVerifierPolicyDecision
    {
        MTPVerifierExecutionPath path =
            MTPVerifierExecutionPath::Unsupported;
        bool accepted_all_position_state_requires_replay = true;
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
        if (supported_sampling_mode &&
            row_local_penalties_supported &&
            input.supports_spec_state_publication)
        {
            decision.path = MTPVerifierExecutionPath::AllPositionStatePublication;
            decision.accepted_all_position_state_requires_replay = false;
            decision.reason =
                input.uses_sampling_penalties
                    ? "greedy_penalties_use_all_position_state_publication"
                    : (input.stochastic_verify
                           ? "stochastic_uses_all_position_state_publication"
                           : "greedy_uses_all_position_state_publication");
            return decision;
        }

        /*
         * This is the Phase 10 grouped-outcome lane. It lets the runner record
         * that grouped verifier math and row-indexed logits are proven, while
         * the orchestrator still decides whether a device-resident publisher is
         * available for the accepted rows. Sampling penalties enter this lane
         * only after the runner proves it can mutate each verifier row with
         * branch-local sampler history.
         */
        if (supported_sampling_mode &&
            row_local_penalties_supported &&
            input.supports_grouped_decode_equivalent_outcome)
        {
            decision.path =
                MTPVerifierExecutionPath::GroupedDecodeEquivalentOutcome;
            decision.accepted_all_position_state_requires_replay = true;
            decision.reason =
                input.uses_sampling_penalties
                    ? "greedy_penalties_use_grouped_decode_equivalent_outcome_with_device_resident_publication"
                    : (input.stochastic_verify
                           ? "stochastic_uses_grouped_decode_equivalent_outcome_with_device_resident_publication"
                           : "greedy_uses_grouped_decode_equivalent_outcome_with_device_resident_publication");
            return decision;
        }

        /*
         * Greedy decode with penalties is still deterministic: the accepted
         * token is the argmax after applying the request-local sparse penalty
         * map. It stays on the shared sequential verifier unless the runner
         * advertises row-local verifier-logit penalty application above.
         */
        const bool use_decode_equivalent_sequential =
            supported_sampling_mode;
        if (use_decode_equivalent_sequential)
        {
            decision.path = MTPVerifierExecutionPath::DecodeEquivalentSequential;
            decision.accepted_all_position_state_requires_replay = true;
            decision.reason = input.uses_sampling_penalties
                                  ? "greedy_penalties_use_shared_decode_equivalent_verifier"
                                  : (input.stochastic_verify
                                         ? "stochastic_uses_shared_decode_equivalent_verifier"
                                         : "greedy_uses_shared_decode_equivalent_verifier");
            return decision;
        }

        decision.path = MTPVerifierExecutionPath::Unsupported;
        decision.accepted_all_position_state_requires_replay = true;
        decision.reason = "sampling_mode_not_supported_by_shared_verifier";
        return decision;
    }

} // namespace llaminar2
