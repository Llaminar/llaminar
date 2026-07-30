#pragma once

#include "execution/prefix_cache/PrefixStateSnapshot.h"
#include "execution/prefix_cache/PrefixCacheStateProbe.h"

#include <string>

namespace llaminar2
{

    struct MTPDecodeStateStamp
    {
        bool valid = false;
        int logical_tokens = 0;
        int main_kv_tokens = 0;
        int shifted_mtp_kv_tokens = 0;
        int position = 0;
        bool has_terminal_hidden = false;
        bool has_terminal_logits = false;
        bool has_ready_token = false;
        PrefixStateProvenance provenance = PrefixStateProvenance::Unknown;
        std::string label;

        bool decodeEquivalent() const
        {
            return isDecodeEquivalent(provenance);
        }
    };

    struct MTPCommitValidationOptions
    {
        bool require_decode_equivalent_source = true;
        bool require_shifted_mtp_kv = true;
        bool require_base_shifted_mtp_kv = true;
        bool require_committed_shifted_mtp_kv = true;
        bool require_terminal_hidden = true;
        bool require_terminal_logits = true;
        bool require_ready_token = true;
    };

    struct MTPStateValidationResult
    {
        bool ok = false;
        std::string reason;

        explicit operator bool() const { return ok; }

        static MTPStateValidationResult success();
        static MTPStateValidationResult failure(std::string reason);
    };

    struct MTPRuntimeSnapshotComparisonOptions
    {
        /**
         * @brief Compare main KV payload hashes in addition to logical cache metadata.
         *
         * Leave this true for production replay checks.  Serial-oracle parity
         * tests may disable it when they compare a row-grouped verifier path
         * against row-by-row decode: the logical KV position must still match,
         * but quantized payload bytes can differ slightly while continuation
         * remains decode-equivalent.
         */
        bool compare_main_kv_payload_hashes = true;
        bool compare_shifted_mtp_kv = true;
        bool compare_gdn_hashes = true;
        bool compare_gdn_values_if_available = false;
        double gdn_relative_l2_tolerance = 1e-4;
        double gdn_max_abs_tolerance = 1e-4;
        double gdn_min_cosine = 0.999999;
    };

    /**
     * @brief Shifted-MTP action required before one serial replay row.
     *
     * A verifier base can already own the shifted row for its first input. In
     * that case replay must reuse the resident row. Once main decode advances,
     * shifted state trails by one and the next replay row must append exactly
     * one shifted entry.
     */
    enum class MTPShiftedReplayRowAction
    {
        ReuseResidentRow,
        AppendRow,
    };

    /**
     * @brief Validated shifted-row plan for a serial verifier replay step.
     */
    struct MTPShiftedReplayRowPlan
    {
        bool ok = false;
        MTPShiftedReplayRowAction action =
            MTPShiftedReplayRowAction::ReuseResidentRow;
        int append_position_offset = -1;
        std::string reason;

        explicit operator bool() const { return ok; }
    };

    /**
     * @brief Serial-visible verifier-state boundary for one MTP decode step.
     *
     * An all-accepted verifier can process one row beyond the final token that
     * the current response budget permits the caller to observe.  That final
     * output token must remain the pending condition token, exactly as it does
     * after ordinary serial decode.  The full verifier outcome remains useful
     * acceptance evidence, but only @ref max_state_commit_rows rows may become
     * live KV, recurrent, terminal-hidden, and logical-position state.
     */
    struct MTPVisibleStateCommitPlan
    {
        bool ok = false;
        int verifier_input_rows = 0;
        int emitted_token_start_index = 0;
        int remaining_output_budget = 0;
        int max_state_commit_rows = 0;
        bool response_boundary_clipped = false;
        std::string reason;

        explicit operator bool() const { return ok; }
    };

    int expectedShiftedMTPTokens(int logical_tokens);

    /**
     * @brief Plan the largest verifier prefix that is serial-visible this step.
     *
     * @param verifier_input_rows Number of target-model rows in the grouped
     *        verifier invocation.
     * @param emitted_token_start_index Number of leading committed output rows
     *        that were already emitted by an earlier transaction.  This is one
     *        for a pending rejection-correction condition and zero otherwise.
     * @param remaining_output_budget Caller-visible output tokens still allowed
     *        for this decode step.  Zero means the caller supplied no finite
     *        response boundary.
     *
     * When the verifier cannot exhaust the finite budget, every accepted row may
     * stay live so its ready token can feed the next transaction.  When it can
     * exhaust the budget, the final visible output remains unprocessed and the
     * live prefix is shortened by exactly one serial row.
     */
    MTPVisibleStateCommitPlan planMTPVisibleStateCommit(
        int verifier_input_rows,
        int emitted_token_start_index,
        int remaining_output_budget);

    /**
     * @brief Decide whether serial replay reuses or appends shifted MTP state.
     *
     * @param main_cached_tokens Current canonical main-model KV count.
     * @param shifted_cached_tokens Current canonical depth-zero MTP KV count.
     * @return A valid reuse/append plan, or a diagnostic failure for every
     *         ambiguous lifecycle shape.
     */
    MTPShiftedReplayRowPlan planMTPShiftedReplayRow(
        int main_cached_tokens,
        int shifted_cached_tokens);

    /**
     * @brief Create a payload-free verifier-base snapshot for the MTP fast path.
     *
     * The all-position publication verifier can skip exporting a second
     * prefix-cache payload when the condition forward has already advanced live
     * state to the verifier base and the sidecar is guaranteed not to mutate
     * main state. This helper creates the narrow logical snapshot used by the
     * transaction validator in that case. It records only token-count
     * invariants; callers must still keep a real rollback checkpoint for
     * failure paths until verifier state is fully isolated in speculative slots.
     */
    PrefixStateSnapshot makeLogicalMTPVerifierBaseSnapshot(int cached_tokens);

    MTPStateValidationResult validateCommittedMTPDecodeState(
        const MTPDecodeStateStamp &state,
        const MTPCommitValidationOptions &options = {});

    MTPStateValidationResult validateAtomicMTPCommit(
        const MTPDecodeStateStamp &base,
        const MTPDecodeStateStamp &committed,
        int emitted_tokens,
        PrefixStateProvenance verifier_source,
        const MTPCommitValidationOptions &options = {});

    MTPStateValidationResult compareMTPRuntimeStateSnapshots(
        const PrefixRuntimeStateSnapshot &oracle,
        const PrefixRuntimeStateSnapshot &candidate,
        const MTPRuntimeSnapshotComparisonOptions &options = {});

} // namespace llaminar2
