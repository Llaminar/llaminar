#pragma once

#include <bit>

/**
 * @file MTPVerifierPolicy.h
 * @brief Production policy for choosing an MTP verifier execution lane.
 *
 * The policy is intentionally fail-closed. Every production MTP verifier uses
 * the grouped decode-equivalent outcome contract; backend-specific publication
 * ownership is selected below this policy without changing verifier semantics.
 * Serial row replay remains a diagnostic/oracle concept owned by focused tests
 * and helper utilities; it is not a production fallback selected here.
 */

namespace llaminar2
{

    /**
     * @brief Immutable policy for mapping logical verifier rows to graph width.
     *
     * Standalone, fixed-depth, and observe-only transactions retain the bounded
     * bucket family because their logical width cannot change behind a captured
     * graph. A scalar dynamic device controller instead owns one maximum-width
     * envelope: active rows are replay data and may change after every accepted
     * transaction without changing graph identity.
     */
    enum class MTPVerifierPhysicalWidthPolicy
    {
        BoundedLogicalBucket,
        DynamicDeviceEnvelope,
    };

    /**
     * @brief Select the bounded physical row width for a logical verifier step.
     *
     * GPU verifier graphs are expensive, immutable executables. Capturing one
     * complete graph for every observed MTP depth retains an unbounded family
     * of nearly identical graphs and can exhaust device memory. This helper
     * defines the one canonical graph-family policy: use power-of-two physical
     * widths and clamp the final family member to the configured maximum.
     *
     * The returned width is a launch/capture property only. Kernels must still
     * consume the exact logical row count from device-resident request state so
     * padded rows cannot update attention, recurrent state, KV cache, routing,
     * sampling, or accepted-state publication.
     *
     * @param logical_rows Number of verifier rows containing real tokens.
     * @param max_rows Largest verifier graph width admitted by configuration.
     * @return A physical width in `[logical_rows, max_rows]`, or zero when the
     *         requested geometry is invalid.
     */
    [[nodiscard]] constexpr int mtpVerifierPhysicalRowBucket(
        int logical_rows,
        int max_rows) noexcept
    {
        if (logical_rows <= 0 || max_rows <= 0 || logical_rows > max_rows)
            return 0;

        const unsigned int rounded =
            std::bit_ceil(static_cast<unsigned int>(logical_rows));
        return rounded <= static_cast<unsigned int>(max_rows)
                   ? static_cast<int>(rounded)
                   : max_rows;
    }

    /**
     * @brief Resolve the physical row stride for a GPU verifier token matrix.
     *
     * Scalar inference is the high-frequency MTP lane and may reuse a bounded
     * power-of-two graph family. Request-batched verification retains its exact
     * logical stride because each following request starts at that stride; a
     * silent bucket there would change every row offset. Keeping this relation
     * in one policy helper prevents callers from interpreting one ambiguous
     * `padded_seq_len` value as both logical state and captured geometry.
     *
     * @param request_count Number of independent verifier requests.
     * @param logical_padded_seq_len Largest real verifier row width.
     * @param max_rows Configured per-request verifier row capacity.
     * @param policy Immutable graph-width ownership policy.
     * @return Physical graph stride, or zero for invalid geometry.
     */
    [[nodiscard]] constexpr int mtpVerifierPhysicalPaddedSeqLen(
        int request_count,
        int logical_padded_seq_len,
        int max_rows,
        MTPVerifierPhysicalWidthPolicy policy) noexcept
    {
        if (request_count <= 0 || logical_padded_seq_len <= 0 ||
            logical_padded_seq_len > max_rows)
        {
            return 0;
        }
        if (policy ==
                MTPVerifierPhysicalWidthPolicy::DynamicDeviceEnvelope &&
            request_count == 1)
        {
            return max_rows;
        }
        return request_count == 1
                   ? mtpVerifierPhysicalRowBucket(
                         logical_padded_seq_len,
                         max_rows)
                   : logical_padded_seq_len;
    }

    enum class MTPVerifierExecutionPath
    {
        Unsupported,
        GroupedDecodeEquivalentOutcome,
    };

    /**
     * @brief Inputs that choose the verifier execution contract for one MTP step.
     *
     * Grouped decode-equivalent verification is a mandatory production
     * contract, not a capability advertisement. If the selected runner later
     * lacks the structural publisher required for its ownership domain, the
     * caller must fail loudly instead of selecting different verifier math or
     * falling back to serial row replay.
     */
    struct MTPVerifierPolicyInput
    {
        bool greedy_sampling = false;
        bool stochastic_verify = false;
        bool uses_sampling_penalties = false;
        bool supports_row_local_penalty_application = false;
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

        /*
         * Route selection below this policy chooses the concrete host-owned or
         * device-owned grouped publisher. Missing grouped infrastructure fails
         * at that boundary; it cannot alter the verifier contract selected here.
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
