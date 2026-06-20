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
     * makes capability reporting precise: a runner can advertise that shared
     * stepwise replay is proven for M=1..4 while direct all-position
     * publication remains disabled for MoE, for example.
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
     * publication is stronger than decode-equivalent replay; callers should
     * require the direct field before publishing state from a batched verifier
     * graph, and fall back to the decode-equivalent field only for the shared
     * one-row replay contract.
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

    /**
     * @brief Phase 9.8 performance capability for one verifier lane.
     *
     * MTPVerifierRowCapability answers "is this verifier row numerically
     * decode-equivalent?".  This object answers the separate production
     * question: "is the proven implementation economical enough to promote?".
     * A serial replay fallback may be correct and still fail this capability.
     * Keeping those states separate prevents dashboards and rollout logic from
     * treating correctness-only paths as vLLM-style fast paths.
     */
    struct MTPVerifierEconomyLane
    {
        bool correct = false;
        bool serial_decode_equivalent_fallback = false;
        bool grouped_decode_equivalent = false;
        bool row_indexed_lm_head = false;
        bool device_resident_input = false;
        bool device_resident_outcome = false;
        bool device_resident_publication = false;
        bool host_bridge_free_hot_path = false;
        bool graph_capturable = false;
        bool greedy = false;
        bool stochastic = false;
        int max_rows = 0;
        std::string perf_gate_status = "unproven";

        static MTPVerifierEconomyLane serialFallbackCorrect(int rows)
        {
            MTPVerifierEconomyLane lane;
            lane.correct = rows > 0;
            lane.serial_decode_equivalent_fallback = lane.correct;
            lane.greedy = lane.correct;
            lane.stochastic = lane.correct;
            lane.max_rows = lane.correct ? rows : 0;
            lane.perf_gate_status = lane.correct
                                        ? "correct_serial_fallback_not_economical"
                                        : "unproven";
            return lane;
        }

        static MTPVerifierEconomyLane groupedPromoted(int rows)
        {
            MTPVerifierEconomyLane lane;
            lane.correct = rows > 0;
            lane.grouped_decode_equivalent = lane.correct;
            lane.row_indexed_lm_head = lane.correct;
            lane.device_resident_input = lane.correct;
            lane.device_resident_outcome = lane.correct;
            lane.device_resident_publication = lane.correct;
            lane.host_bridge_free_hot_path = lane.correct;
            lane.graph_capturable = lane.correct;
            lane.greedy = lane.correct;
            lane.stochastic = lane.correct;
            lane.max_rows = lane.correct ? rows : 0;
            lane.perf_gate_status = lane.correct
                                        ? "grouped_promoted"
                                        : "unproven";
            return lane;
        }

        /**
         * @brief Report a grouped verifier outcome proof with resident
         *        publication, but without accepting the full hot-path economy.
         *
         * This is the important Phase 10 middle state for GPU MoE: focused
         * routed/shared verifier kernels prove strict M=2..4 row equivalence,
         * the runner can publish accepted rows from device-resident outcome
         * metadata, but the full verifier graph has not yet met the MTP speed
         * target. Keeping this state first-class stops future policy code from
         * conflating "batched verifier math and publication are correct" with
         * "the whole MTP transaction is economical".
         */
        static MTPVerifierEconomyLane groupedOutcomeDevicePublicationEconomicsPending(
            int rows)
        {
            MTPVerifierEconomyLane lane;
            lane.correct = rows > 0;
            lane.serial_decode_equivalent_fallback = lane.correct;
            lane.grouped_decode_equivalent = lane.correct;
            lane.row_indexed_lm_head = lane.correct;
            lane.device_resident_input = lane.correct;
            lane.device_resident_outcome = lane.correct;
            lane.device_resident_publication = lane.correct;
            lane.host_bridge_free_hot_path = false;
            lane.graph_capturable = lane.correct;
            lane.greedy = lane.correct;
            lane.stochastic = lane.correct;
            lane.max_rows = lane.correct ? rows : 0;
            lane.perf_gate_status = lane.correct
                                        ? "grouped_outcome_economics_pending"
                                        : "unproven";
            return lane;
        }

        bool supportsRows(
            int rows,
            bool stochastic_requested = false) const
        {
            if (!correct || rows <= 0 || rows > max_rows)
                return false;
            return stochastic_requested ? stochastic : greedy;
        }

        bool isEconomicalForRows(
            int rows,
            bool stochastic_requested = false) const
        {
            return supportsRows(rows, stochastic_requested) &&
                   grouped_decode_equivalent &&
                   row_indexed_lm_head &&
                   device_resident_input &&
                   device_resident_outcome &&
                   device_resident_publication &&
                   host_bridge_free_hot_path &&
                   graph_capturable;
        }

        void intersectWith(const MTPVerifierEconomyLane &other)
        {
            correct = correct && other.correct;
            max_rows = correct ? std::min(max_rows, other.max_rows) : 0;
            serial_decode_equivalent_fallback =
                serial_decode_equivalent_fallback &&
                other.serial_decode_equivalent_fallback;
            grouped_decode_equivalent =
                grouped_decode_equivalent &&
                other.grouped_decode_equivalent;
            row_indexed_lm_head = row_indexed_lm_head && other.row_indexed_lm_head;
            device_resident_input =
                device_resident_input && other.device_resident_input;
            device_resident_outcome =
                device_resident_outcome && other.device_resident_outcome;
            device_resident_publication =
                device_resident_publication && other.device_resident_publication;
            host_bridge_free_hot_path =
                host_bridge_free_hot_path && other.host_bridge_free_hot_path;
            graph_capturable = graph_capturable && other.graph_capturable;
            greedy = greedy && other.greedy;
            stochastic = stochastic && other.stochastic;
            if (!correct)
            {
                perf_gate_status = "unproven";
            }
            else if (perf_gate_status != other.perf_gate_status)
            {
                perf_gate_status = "mixed_capability";
            }
        }
    };

    /**
     * @brief Phase 9.8 verifier economy matrix by model family.
     */
    struct MTPVerifierEconomyCapability
    {
        MTPVerifierEconomyLane dense;
        MTPVerifierEconomyLane moe;

        bool supportsDenseRows(
            int rows,
            bool stochastic_requested = false) const
        {
            return dense.supportsRows(rows, stochastic_requested);
        }

        bool supportsMoERows(
            int rows,
            bool stochastic_requested = false) const
        {
            return moe.supportsRows(rows, stochastic_requested);
        }

        bool hasEconomicalDensePath(
            int rows,
            bool stochastic_requested = false) const
        {
            return dense.isEconomicalForRows(rows, stochastic_requested);
        }

        bool hasEconomicalMoEPath(
            int rows,
            bool stochastic_requested = false) const
        {
            return moe.isEconomicalForRows(rows, stochastic_requested);
        }

        void intersectWith(const MTPVerifierEconomyCapability &other)
        {
            dense.intersectWith(other.dense);
            moe.intersectWith(other.moe);
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
