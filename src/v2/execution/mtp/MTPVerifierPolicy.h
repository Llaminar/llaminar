#pragma once

namespace llaminar2
{

    enum class MTPVerifierExecutionPath
    {
        Unsupported,
        AllPositionStatePublication,
        DecodeEquivalentSequential,
    };

    struct MTPVerifierPolicyInput
    {
        bool greedy_sampling = false;
        bool stochastic_verify = false;
        bool uses_sampling_penalties = false;
        bool supports_spec_state_publication = false;
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
        if (supported_sampling_mode &&
            !input.uses_sampling_penalties &&
            input.supports_spec_state_publication)
        {
            decision.path = MTPVerifierExecutionPath::AllPositionStatePublication;
            decision.accepted_all_position_state_requires_replay = false;
            decision.reason = input.stochastic_verify
                                  ? "stochastic_uses_all_position_state_publication"
                                  : "greedy_uses_all_position_state_publication";
            return decision;
        }

        /*
         * Greedy decode with penalties is still deterministic: the accepted
         * token is the argmax after applying the request-local sparse penalty
         * map.  It cannot use the all-position publication shortcut until that
         * shortcut applies the correct row-local penalty history, but the
         * shared sequential verifier already samples each target and draft row
         * after applying the same penalties as normal decode.
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
