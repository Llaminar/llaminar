/**
 * @file MTPVerifierPolicy.h
 * @brief MTP transaction budgets, physical verifier width and execution contracts.
 *
 * Physical width follows immutable execution ownership: hosted tickets select
 * exact retained row identities, native dynamic parents share one maximum
 * envelope, and standalone verification uses bounded buckets. Device-owned
 * live counts still clip useful rows and publication within the selected graph.
 *
 * The policy is intentionally fail-closed. Every production MTP verifier uses
 * the grouped decode-equivalent outcome contract; backend-specific publication
 * ownership is selected below this policy without changing verifier semantics.
 * Serial row replay remains a diagnostic/oracle concept owned by focused tests
 * and helper utilities; it is not a production fallback selected here.
 */
#pragma once

#include "MTPVerifierGraphFamily.h"
#include "MTPVerifierPhysicalWidthPolicy.h"
#include <bit>
#include <algorithm>

namespace llaminar2
{

    /** @brief Owner that clips speculative work at a response boundary. */
    enum class MTPCommitBudgetAuthority
    {
        Host,
        Device,
    };

    /**
     * @brief Resolve executed drafts once, before checkpoint admission.
     *
     * A device controller retains the complete selected transaction when any
     * speculative response work remains; its commit budget clips publication,
     * not graph width. A zero-draft response boundary executes only the existing
     * captured condition advance. The same value must size rollback protection
     * and drive the subsequent sidecar/verifier execution.
     *
     * @param capacity Draft width selected for this transaction/capture.
     * @param response_budget Positive output limit, or zero for unbounded.
     * @param leading_output_cost Zero for an already-emitted condition, else one.
     * @param authority Owner of accepted-state/output clipping.
     * @return Executed draft width, or -1 for invalid admission geometry.
     */
    [[nodiscard]] constexpr int mtpTransactionDraftCount(
        int capacity, int response_budget, int leading_output_cost,
        MTPCommitBudgetAuthority authority) noexcept
    {
        if (capacity < 0 || response_budget < 0 ||
            leading_output_cost < 0 || leading_output_cost > 1)
            return -1;
        if (response_budget == 0)
            return capacity;
        const int remaining = std::max(0, response_budget - leading_output_cost);
        if (remaining == 0)
            return 0;
        return authority == MTPCommitBudgetAuthority::Device
                   ? capacity : std::min(capacity, remaining);
    }

    /**
     * @brief Select the bounded physical row width for a logical verifier step.
     *
     * This is the bounded family's mapping: power-of-two physical widths,
     * clamped at admitted capacity. Hosted exact families use ExactLogicalRows
     * instead and pay for every retained identity during physical admission.
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
        switch (policy)
        {
        case MTPVerifierPhysicalWidthPolicy::DynamicDeviceEnvelope:
            return request_count == 1 ? max_rows : logical_padded_seq_len;
        case MTPVerifierPhysicalWidthPolicy::BoundedLogicalBucket:
            return request_count == 1
                ? mtpVerifierPhysicalRowBucket(logical_padded_seq_len, max_rows)
                : logical_padded_seq_len;
        case MTPVerifierPhysicalWidthPolicy::ExactLogicalRows:
            return logical_padded_seq_len;
        }
        return 0; // Unknown ownership must never select an executable.
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
