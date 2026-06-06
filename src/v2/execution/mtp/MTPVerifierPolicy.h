#pragma once

namespace llaminar2
{

    enum class MTPVerifierExecutionPath
    {
        AllPositionVerifier,
        DecodeEquivalentSequential,
    };

    struct MTPVerifierPolicyInput
    {
        bool runner_requires_decode_equivalent_replay = false;
        bool runner_supports_verifier_state_row_restore = false;
        bool greedy_sampling = false;
        bool stochastic_verify = false;
        bool uses_sampling_penalties = false;
        bool disable_decode_equivalent_sequential = false;
    };

    struct MTPVerifierPolicyDecision
    {
        MTPVerifierExecutionPath path =
            MTPVerifierExecutionPath::AllPositionVerifier;
        bool allow_verifier_state_row_shortcut = false;
        bool accepted_all_position_state_requires_replay = true;
        const char *reason = "verifier_prefill_rows_not_proven_decode_equivalent";
    };

    inline MTPVerifierPolicyDecision chooseMTPVerifierPolicy(
        const MTPVerifierPolicyInput &input)
    {
        MTPVerifierPolicyDecision decision;

        const bool can_use_decode_equivalent_sequential =
            input.greedy_sampling &&
            !input.stochastic_verify &&
            !input.uses_sampling_penalties &&
            !input.disable_decode_equivalent_sequential;

        if (input.runner_requires_decode_equivalent_replay &&
            can_use_decode_equivalent_sequential)
        {
            decision.path = MTPVerifierExecutionPath::DecodeEquivalentSequential;
            decision.allow_verifier_state_row_shortcut = false;
            decision.accepted_all_position_state_requires_replay = true;
            decision.reason = "stateful_runner_requires_decode_equivalent_replay";
            return decision;
        }

        decision.path = MTPVerifierExecutionPath::AllPositionVerifier;
        decision.allow_verifier_state_row_shortcut =
            input.runner_supports_verifier_state_row_restore &&
            !input.runner_requires_decode_equivalent_replay;
        decision.accepted_all_position_state_requires_replay =
            !decision.allow_verifier_state_row_shortcut;

        if (input.runner_requires_decode_equivalent_replay)
        {
            decision.reason = input.disable_decode_equivalent_sequential
                                  ? "decode_equivalent_replay_debug_override"
                                  : "decode_equivalent_replay_unavailable_for_sampling_mode";
        }
        else if (!input.runner_supports_verifier_state_row_restore)
        {
            decision.reason = "verifier_state_row_restore_not_supported";
        }
        else
        {
            decision.reason = "verifier_prefill_rows_declared_decode_equivalent";
        }

        return decision;
    }

} // namespace llaminar2
