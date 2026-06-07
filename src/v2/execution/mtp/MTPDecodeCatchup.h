/**
 * @file MTPDecodeCatchup.h
 * @brief Shared decode-equivalent MTP catch-up contract.
 */
#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace llaminar2
{
    class IInferenceRunner;
    struct PrefixStateSnapshot;

    struct MTPDecodeCatchupGreedyRequest
    {
        std::vector<int32_t> draft_tokens;
        std::vector<int32_t> stop_tokens;
        int base_sidecar_position = 0;
        bool allow_speculative_discard = true;
        std::string verifier_path = "decode_equivalent_catchup";
        std::string implementation_name = "shared_stepwise";

        /**
         * Optional decode-equivalent verifier base captured before sidecar
         * drafting. Optimized hooks that discover a rejection after a batched
         * verifier attempt must restore this exact base before replaying the
         * correction path; a post-sidecar checkpoint can already contain
         * shifted-MTP cache mutations.
         */
        const PrefixStateSnapshot *verifier_base_checkpoint = nullptr;
    };

    struct MTPDecodeCatchupGreedyResult
    {
        bool ok = false;
        std::string error;

        std::vector<int32_t> accepted_tokens;
        std::vector<int32_t> verifier_tokens;

        bool all_speculative_accepted = true;
        bool stopped_on_output = false;
        int accepted_speculative_prefix = 0;
        int32_t rejected_verified_token = -1;
        int32_t ready_token = -1;

        int main_forward_token_count = 0;
        int shifted_commit_count = 0;
        std::string debug_trace;

        /**
         * Number of verifier input rows whose target-model state may be
         * published directly.
         *
         * Stepwise catch-up forwards every committed output token, so the
         * default (-1) means "same as accepted_tokens.size()". A vLLM-style
         * verifier graph is different after a rejection: the correction token
         * is sampled from the rejecting row, but that correction token has not
         * itself been forwarded. Such candidates set this to the accepted
         * verifier-input prefix and replay the correction suffix before
         * claiming decode equivalence.
         */
        int target_verifier_state_commit_count = -1;
    };

    struct MTPDecodeCatchupGreedyEquivalence
    {
        bool ok = false;
        std::string error;
    };

    using MTPDecodeCatchupGreedySampler = std::function<int32_t()>;

    MTPDecodeCatchupGreedyEquivalence compareMTPDecodeCatchupGreedyResults(
        const MTPDecodeCatchupGreedyResult &oracle,
        const MTPDecodeCatchupGreedyResult &candidate);

    /**
     * @brief Build the greedy catch-up result implied by target verifier rows.
     *
     * sampled_verifier_tokens has one row per verifier input token:
     *   row i is the target sample after forwarding request.draft_tokens[i].
     *
     * This is intentionally not the same as stepwise catch-up after a reject.
     * The rejecting row supplies a correction output token, but the live target
     * state is only valid through the accepted verifier-input prefix. Callers
     * must replay any correction suffix before publishing state as fully
     * decode-equivalent.
     */
    MTPDecodeCatchupGreedyResult buildMTPDecodeCatchupGreedyResultFromVerifierRows(
        const MTPDecodeCatchupGreedyRequest &request,
        const std::vector<int32_t> &sampled_verifier_tokens);

    /**
     * @brief Run greedy MTP verification through normal one-token decode.
     *
     * This is the canonical shared catch-up implementation for stateful models
     * where batched all-position verifier rows are not known to leave mutable
     * KV/GDN/decode state equal to stepwise decode. CUDA and ROCm optimized
     * catch-up implementations must prove equivalence against this contract
     * before they are promoted.
     */
    MTPDecodeCatchupGreedyResult runSharedStepwiseMTPDecodeCatchupGreedy(
        IInferenceRunner &runner,
        const MTPDecodeCatchupGreedyRequest &request,
        const MTPDecodeCatchupGreedySampler &sample_after_forward);

} // namespace llaminar2
