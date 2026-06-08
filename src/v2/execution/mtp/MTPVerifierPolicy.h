#pragma once

namespace llaminar2
{

    enum class MTPVerifierExecutionPath
    {
        Unsupported,
        DecodeEquivalentSequential,
    };

    struct MTPVerifierPolicyInput
    {
        bool greedy_sampling = false;
        bool stochastic_verify = false;
        bool uses_sampling_penalties = false;
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

        const bool use_decode_equivalent_sequential =
            (input.greedy_sampling || input.stochastic_verify) &&
            !input.uses_sampling_penalties;
        if (use_decode_equivalent_sequential)
        {
            decision.path = MTPVerifierExecutionPath::DecodeEquivalentSequential;
            decision.accepted_all_position_state_requires_replay = true;
            decision.reason = input.stochastic_verify
                                  ? "stochastic_uses_shared_decode_equivalent_verifier"
                                  : "greedy_uses_shared_decode_equivalent_verifier";
            return decision;
        }

        decision.path = MTPVerifierExecutionPath::Unsupported;
        decision.accepted_all_position_state_requires_replay = true;
        decision.reason = input.uses_sampling_penalties
                              ? "greedy_penalty_mtp_requires_new_transaction_path"
                              : "sampling_mode_not_supported_by_shared_verifier";
        return decision;
    }

} // namespace llaminar2
