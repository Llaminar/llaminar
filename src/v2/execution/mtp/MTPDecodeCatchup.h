/**
 * @file MTPDecodeCatchup.h
 * @brief Shared decode-equivalent MTP catch-up contract.
 */
#pragma once

#include <algorithm>
#include <cstdint>
#include <functional>
#include <optional>
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
     * verifier-input prefix. Runners may either replay the correction suffix
     * immediately to produce a ready token or defer it as the next ordinary
     * condition token, matching the vLLM speculative-state publication shape.
         */
        int target_verifier_state_commit_count = -1;
    };

    /**
     * @brief Result for a compact batch of all-position verifier catch-up rows.
     *
     * The rows are still interpreted per request by the single-request
     * `MTPDecodeCatchupGreedyResult` contract.  Keeping the outer batch result
     * thin makes failures easy to report while preserving request-local
     * equivalence checks.
     */
    struct MTPDecodeCatchupGreedyBatchResult
    {
        bool ok = false;
        std::string error;
        std::vector<MTPDecodeCatchupGreedyResult> results;
    };

    struct MTPDecodeCatchupGreedyEquivalence
    {
        bool ok = false;
        std::string error;
    };

    using MTPDecodeCatchupGreedySampler = std::function<int32_t(int32_t forwarded_token)>;

    /**
     * @brief Numeric proof contract for one verifier-row implementation.
     *
     * Phase 9.7 promotes verifier implementations only by model family, row
     * count, and sampling mode.  Keeping those facts in a plain value type
     * makes equivalence evidence precise: a runner can record the contiguous
     * runtime-M range proven for a model family and sampling mode.
     */
    struct MTPVerifierRowEquivalenceSpec
    {
        bool enabled = false;
        int max_rows = 0;
        bool greedy = false;
        bool stochastic = false;
        double min_cosine = 0.99995;
        double max_rel_l2 = 0.005;
        double max_symmetric_kl = 1.0e-4;

        static MTPVerifierRowEquivalenceSpec proven(
            int rows,
            bool supports_stochastic = true)
        {
            MTPVerifierRowEquivalenceSpec spec;
            spec.enabled = rows > 0;
            spec.max_rows = rows > 0 ? rows : 0;
            spec.greedy = rows > 0;
            spec.stochastic = rows > 0 && supports_stochastic;
            return spec;
        }

        bool supportsRows(int rows, bool stochastic_requested = false) const
        {
            if (!enabled || rows <= 0 || rows > max_rows)
            {
                return false;
            }
            return stochastic_requested ? stochastic : greedy;
        }

        void intersectWith(const MTPVerifierRowEquivalenceSpec &other)
        {
            enabled = enabled && other.enabled;
            max_rows = enabled ? std::min(max_rows, other.max_rows) : 0;
            greedy = greedy && other.greedy;
            stochastic = stochastic && other.stochastic;
            min_cosine = std::max(min_cosine, other.min_cosine);
            max_rel_l2 = std::min(max_rel_l2, other.max_rel_l2);
            max_symmetric_kl = std::min(max_symmetric_kl, other.max_symmetric_kl);
        }
    };

    /**
     * @brief Runner-level verifier-row capability matrix.
     *
     * The dense and MoE fields are intentionally separate because their fast
     * paths advance different mutable state surfaces.  Direct all-position
     * publication is stronger than decode-equivalent grouped verifier rows;
     * callers should require the direct field before publishing state from a
     * batched verifier graph.  Decode-equivalent row support proves the grouped
     * verifier math, not permission to run production row replay.
     */
    struct MTPVerifierRowCapability
    {
        MTPVerifierRowEquivalenceSpec dense_decode_equivalent;
        MTPVerifierRowEquivalenceSpec moe_decode_equivalent;
        MTPVerifierRowEquivalenceSpec dense_direct_all_position;
        MTPVerifierRowEquivalenceSpec moe_direct_all_position;
        bool device_resident_direct_publication = false;

        bool supportsDenseDecodeEquivalentRows(
            int rows,
            bool stochastic_requested = false) const
        {
            return dense_decode_equivalent.supportsRows(
                rows,
                stochastic_requested);
        }

        bool supportsMoEDecodeEquivalentRows(
            int rows,
            bool stochastic_requested = false) const
        {
            return moe_decode_equivalent.supportsRows(
                rows,
                stochastic_requested);
        }

        bool supportsDenseDirectAllPositionRows(
            int rows,
            bool stochastic_requested = false) const
        {
            return dense_direct_all_position.supportsRows(
                rows,
                stochastic_requested);
        }

        bool supportsMoEDirectAllPositionRows(
            int rows,
            bool stochastic_requested = false) const
        {
            return moe_direct_all_position.supportsRows(
                rows,
                stochastic_requested);
        }

        void intersectWith(const MTPVerifierRowCapability &other)
        {
            dense_decode_equivalent.intersectWith(other.dense_decode_equivalent);
            moe_decode_equivalent.intersectWith(other.moe_decode_equivalent);
            dense_direct_all_position.intersectWith(other.dense_direct_all_position);
            moe_direct_all_position.intersectWith(other.moe_direct_all_position);
            device_resident_direct_publication =
                device_resident_direct_publication &&
                other.device_resident_direct_publication;
        }
    };

    MTPDecodeCatchupGreedyEquivalence compareMTPDecodeCatchupGreedyResults(
        const MTPDecodeCatchupGreedyResult &oracle,
        const MTPDecodeCatchupGreedyResult &candidate);

    /**
     * @brief Build the greedy catch-up contract from one all-position verifier
     * forward.
     *
     * The verifier rows are the sampled target-model tokens for each input row
     * in request.draft_tokens. Row 0 verifies request.draft_tokens[1], row N-2
     * verifies request.draft_tokens[N-1], and row N-1 is the bonus ready token
     * when every speculative token is accepted.
     *
     * After a rejection, the correcting token is sampled from the rejecting
     * row but has not itself been forwarded by the all-position verifier. The
     * result therefore publishes only the accepted verifier-input prefix and
     * leaves target_verifier_state_commit_count smaller than
     * accepted_tokens.size(). A caller that has already replayed the correction
     * token can pass correction_replay_ready_token to make the token stream
     * fully comparable with stepwise decode.
     */
    MTPDecodeCatchupGreedyResult buildAllPositionMTPDecodeCatchupGreedyResult(
        const MTPDecodeCatchupGreedyRequest &request,
        const std::vector<int32_t> &sampled_verifier_rows,
        std::optional<int32_t> correction_replay_ready_token = std::nullopt);

    /**
     * @brief Build greedy catch-up results from one compact verifier row batch.
     *
     * Rows are consumed in request order, with exactly
     * `request.draft_tokens.size()` rows assigned to each request.  This is the
     * CPU-side mirror of the graph row materializer: graph execution may be
     * padded, but the sampled-row vector passed here must already be compact.
     */
    MTPDecodeCatchupGreedyBatchResult buildAllPositionMTPDecodeCatchupGreedyBatchResult(
        const std::vector<MTPDecodeCatchupGreedyRequest> &requests,
        const std::vector<int32_t> &sampled_verifier_rows,
        const std::vector<std::optional<int32_t>> &correction_replay_ready_tokens = {});

    /**
     * @brief Run greedy MTP verification through normal one-token decode.
     *
     * This is the canonical shared catch-up implementation for stateful models
     * where batched all-position verifier rows are not known to leave mutable
     * KV/GDN/decode state equal to stepwise decode. CUDA and ROCm optimized
     * catch-up implementations must prove equivalence against this contract
     * before they are promoted.
     *
     * The sampler callback receives the token that was just forwarded.  Penalty
     * aware callers use that value to update their branch-local sampler history
     * before sampling the row's verifier logits.
     */
    MTPDecodeCatchupGreedyResult runSharedStepwiseMTPDecodeCatchupGreedy(
        IInferenceRunner &runner,
        const MTPDecodeCatchupGreedyRequest &request,
        const MTPDecodeCatchupGreedySampler &sample_after_forward);

} // namespace llaminar2
