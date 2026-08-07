/**
 * @file MTPVerifierOutcomeGraph.h
 * @brief Typed policy and stable device bindings for graph-owned MTP outcomes.
 *
 * A grouped verifier forward does not end when the LM head writes logits.  The
 * complete GPU transaction also selects verifier tokens and reduces the
 * accepted prefix into a compact outcome.  In mirrored LocalTP execution every
 * participant performs that same terminal transaction against its own
 * byte-identical full-vocabulary logits.  This header describes both the
 * operation and its ownership without depending on a particular model graph.
 *
 * All pointers in MTPVerifierOutcomeGraphBinding refer to persistent arena
 * storage.  They are installed before any graph is built and remain stable for
 * the lifetime of every cached graph that captures them.  Request-specific
 * values are written into those buffers before replay; graph stages must never
 * capture pointers to temporary host vectors or per-request allocations.
 */

#pragma once

#include <cstdint>

namespace llaminar2
{
    /**
     * @brief Immutable presence/frequency policy admitted once per request.
     *
     * These values originate in the serving request and may therefore cross the
     * host/device boundary at request admission.  They must never be rebuilt or
     * republished from a speculative transaction: doing so would make captured
     * graph identity depend on mutable host control flow.
     *
     * The device-only `first_token_already_in_history` state is intentionally
     * absent.  It is produced by accepted-state publication and lives in
     * @ref MTPGreedyPenaltyPolicy behind a graph-stable arena address.
     */
    struct MTPRequestPenaltyPolicy
    {
        float presence_penalty = 0.0f;
        float frequency_penalty = 0.0f;

        /** @return Whether serial decode applies either history penalty. */
        [[nodiscard]] bool enabled() const noexcept
        {
            return presence_penalty != 0.0f || frequency_penalty != 0.0f;
        }

        bool operator==(const MTPRequestPenaltyPolicy &) const = default;
    };

    /**
     * @brief Request policy consumed by graph-owned grouped greedy sampling.
     *
     * Presence and frequency penalties are part of the decode algorithm, not a
     * post-processing step.  Immutable magnitudes are copied from
     * @ref MTPRequestPenaltyPolicy once at request admission.  The
     * `first_token_already_in_history` member is different: accepted-state
     * publication updates it on device after every compact outcome.  Captured
     * proposal and verifier kernels only read this resident structure; no graph
     * node accepts a host-authored transaction bit.
     */
    struct alignas(16) MTPGreedyPenaltyPolicy
    {
        float presence_penalty = 0.0f;
        float frequency_penalty = 0.0f;
        int32_t first_token_already_in_history = 0;
        int32_t enabled = 0;
    };

    static_assert(sizeof(MTPGreedyPenaltyPolicy) == 4 * sizeof(int32_t));
    constexpr int kMTPGreedyPenaltyPolicyWords =
        static_cast<int>(sizeof(MTPGreedyPenaltyPolicy) / sizeof(int32_t));

    /**
     * @brief Verifier outcome operation captured at the end of a forward graph.
     *
     * The enum is deliberately closed.  A new sampling regime must add a real
     * graph implementation and a cache-signature value rather than quietly
     * sharing a topology intended for another verifier.
     */
    enum class MTPVerifierOutcomeGraphMode : uint8_t
    {
        Disabled = 0,
        Greedy = 1,
        StochasticSerialEquivalent = 2,
        StochasticRejection = 3,
    };

    /**
     * @brief Declarative owner of graph-captured verifier outcome reduction.
     *
     * The policy is deliberately independent of the sampling mode.  A model
     * graph declares where terminal reduction lives, while reusable graph
     * machinery wires the corresponding kernels and event publication.  This
     * prevents topology-sensitive collectives from being inferred deep inside
     * a compute stage.
     */
    enum class MTPVerifierOutcomeOwnershipPolicy : uint8_t
    {
        /** No ownership contract was declared; graph construction must fail. */
        Unspecified = 0,

        /**
         * Every graph participant owns and publishes its local compact result.
         *
         * SingleDevice naturally has one owner.  Mirrored LocalTP requires a
         * full-vocabulary, byte-identical head on every participant; each child
         * runs identical reduction math and publishes readiness on its exact
         * graph stream.  No rank outcome broadcast is part of this policy.
         */
        ParticipantLocal = 1,
    };

    /**
     * @brief Persistent device storage consumed by the verifier outcome stage.
     *
     * The binding contains addresses only, never request values.  For example,
     * @ref stop_tokens_device always addresses an eight-element arena row; the
     * pre-replay metadata transaction updates that row and fills unused entries
     * with `-1`.  The captured reducer can consequently inspect all eight slots
     * without embedding a changing host stop-token count in its kernel node.
     */
    struct MTPVerifierOutcomeGraphBinding
    {
        const int32_t *verifier_input_tokens_device = nullptr;
        const int32_t *active_verifier_row_count_device = nullptr;
        /** Prepared controller budget for the current verifier transaction. */
        const uint32_t *transaction_commit_budget_device = nullptr;
        /**
         * Controller-owned carry count for the transaction being summarized.
         *
         * A value of one means output row zero was already committed by the
         * preceding transaction boundary.  This is deliberately distinct from
         * the greedy penalty policy's history-membership bit: sampling history
         * and response-ledger ownership advance at different lifecycle edges.
         */
        const int32_t *next_leading_committed_output_count_device = nullptr;
        const int32_t *stop_tokens_device = nullptr;
        const MTPGreedyPenaltyPolicy *penalty_policy_device = nullptr;
        int32_t *generated_token_counts_device = nullptr;
        int generated_token_count_capacity = 0;

        int32_t *verifier_tokens_device = nullptr;
        float *argmax_values_device = nullptr;
        float *argmax_partial_values_device = nullptr;
        int32_t *argmax_partial_indices_device = nullptr;
        int argmax_partial_capacity = 0;

        int32_t *output_tokens_device = nullptr;
        int32_t *output_meta_device = nullptr;
        int output_token_capacity = 0;
        int output_meta_capacity = 0;

        /**
         * @brief Validate the pointer/capacity contract required by greedy mode.
         */
        [[nodiscard]] bool validForGreedy() const noexcept
        {
            return verifier_input_tokens_device != nullptr &&
                   active_verifier_row_count_device != nullptr &&
                   transaction_commit_budget_device != nullptr &&
                   next_leading_committed_output_count_device != nullptr &&
                   stop_tokens_device != nullptr &&
                   penalty_policy_device != nullptr &&
                   generated_token_counts_device != nullptr &&
                   generated_token_count_capacity > 0 &&
                   verifier_tokens_device != nullptr &&
                   argmax_values_device != nullptr &&
                   argmax_partial_values_device != nullptr &&
                   argmax_partial_indices_device != nullptr &&
                   argmax_partial_capacity > 0 &&
                   output_tokens_device != nullptr &&
                   output_meta_device != nullptr &&
                   output_token_capacity > 0 &&
                   output_meta_capacity > 0;
        }
    };

} // namespace llaminar2
