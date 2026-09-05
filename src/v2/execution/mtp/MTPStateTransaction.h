/**
 * @file MTPStateTransaction.h
 * @brief Typed validation contracts for persistent MTP and prefix-replay state.
 *
 * The declarations in this file define the legal equivalence boundaries for
 * MTP publication, rollback, and prefix restoration.  Exact payload identity
 * is the default.  A caller may request placement-aware numerical comparison
 * only when the snapshots themselves prove that an ExpertOverlay movement
 * epoch occurred and retain every native-precision value needed to certify the
 * result.
 */

#pragma once

#include "execution/prefix_cache/PrefixStateSnapshot.h"
#include "execution/prefix_cache/PrefixCacheStateProbe.h"

#include <optional>
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

        /**
         * @brief Numerical evidence for one placement-aware terminal payload.
         *
         * Exact byte comparisons leave @ref compared false.  A true value
         * means full retained FP32 payloads were compared after the snapshots
         * proved that their MoE placement epochs differ.
         */
        struct TerminalPayloadNumericalEvidence
        {
            bool compared = false;
            bool passed = false;
            size_t elements = 0;
            double cosine = 0.0;
            double relative_l2 = 0.0;
            double max_abs = 0.0;
        };

        /**
         * @brief Aggregate proof for recomputed main-KV suffix payloads.
         *
         * A partial prefix hit restores its cached prefix byte-for-byte but
         * recomputes the uncached suffix.  If ExpertOverlay changed the
         * executing device type between the serial oracle and replay, only
         * that suffix may differ numerically.  This record aggregates the
         * strict prefix hashes and full suffix-value comparisons across every
         * affected layer and both K/V payloads.
         */
        struct MainKVNumericalEvidence
        {
            bool compared = false;
            bool passed = false;
            size_t exact_prefix_segments = 0;
            size_t numerical_suffix_payloads = 0;
            size_t elements = 0;
            double minimum_cosine = 1.0;
            double maximum_relative_l2 = 0.0;
            double maximum_abs = 0.0;
        };

        /**
         * @brief Aggregate proof for numerically compared GDN live state.
         *
         * Exact hashes remain the default.  This evidence is populated only
         * when a typed comparison policy requires every retained recurrence
         * and short-convolution value to be materialized and compared.
         */
        struct GDNStateNumericalEvidence
        {
            bool compared = false;
            bool passed = false;
            size_t payloads = 0;
            size_t elements = 0;
            double minimum_cosine = 1.0;
            double maximum_relative_l2 = 0.0;
            double maximum_abs = 0.0;
        };

        /// Evidence for the persistent final hidden row.
        TerminalPayloadNumericalEvidence terminal_hidden_numerical;

        /// Evidence for the persistent final logits row.
        TerminalPayloadNumericalEvidence terminal_logits_numerical;

        /// Evidence for placement-aware main-KV suffix recomputation.
        MainKVNumericalEvidence main_kv_numerical;

        /// Evidence for placement-aware or explicitly numerical GDN state.
        GDNStateNumericalEvidence gdn_numerical;

        explicit operator bool() const { return ok; }

        static MTPStateValidationResult success();
        static MTPStateValidationResult failure(std::string reason);
    };

    /**
     * @brief Contract used to compare a persistent FP32 terminal payload.
     */
    enum class MTPTerminalPayloadComparisonPolicy
    {
        /** Require identical byte counts and hashes in every circumstance. */
        ExactBytes,

        /**
         * Require exact bytes while placement is unchanged; after a proven
         * MoE movement epoch, require full-payload numerical equivalence.
         */
        ExactUnlessMoEPlacementChanged,
    };

    /**
     * @brief Legal comparison policies for persistent main-model KV payloads.
     */
    enum class MTPMainKVPayloadComparisonPolicy
    {
        /** Require the complete canonical K/V payload to be byte-identical. */
        ExactBytes,

        /**
         * Require the restored prefix segment to remain byte-identical.  After
         * a proven ExpertOverlay movement epoch only, allow the bounded
         * recomputed suffix segment to satisfy a full-value numerical gate.
         */
        ExactPrefixNumericalSuffixAfterMoEPlacementChange,
    };

    /**
     * @brief Legal comparison contracts for persistent GDN live state.
     */
    enum class MTPGDNStateComparisonPolicy
    {
        /** Compare geometry only; used by continuation-based diagnostics. */
        LogicalMetadataOnly,

        /** Require identical recurrence and short-convolution bytes. */
        ExactBytes,

        /**
         * Compare every retained FP32 value against the configured numerical
         * thresholds regardless of hash identity.  This is intended for
         * focused serial/grouped publication diagnostics.
         */
        NumericalValues,

        /**
         * Require exact bytes while placement is unchanged.  After the
         * snapshots prove a movement-epoch transition, compare every live GDN
         * value numerically because the recomputed suffix can inherit a
         * backend- or participant-order rounding difference from MoE.
         */
        ExactUnlessMoEPlacementChanged,
    };

    /**
     * @brief Default full-row cosine floor after a cross-device expert move.
     *
     * Callers may require a stricter model-specific threshold, but a
     * placement-aware state comparison must never silently inherit a loose
     * token-level gate.
     */
    inline constexpr double
        kDefaultPlacementAwareTerminalPayloadMinimumCosine = 0.99;

    /** Strict default cosine floor for a recomputed KV suffix row. */
    inline constexpr double
        kDefaultPlacementAwareMainKVSuffixMinimumCosine = 0.99;

    /** Cosine floor for a complete GDN bank after placement changed. */
    inline constexpr double
        kDefaultPlacementAwareGDNMinimumCosine = 0.99;

    /** Scale-aware error ceiling for a complete placement-aware GDN bank. */
    inline constexpr double
        kDefaultPlacementAwareGDNMaximumRelativeL2 = 0.05;

    /**
     * @brief Options for comparing two authenticated runtime-state snapshots.
     *
     * Every terminal payload defaults to exact identity.  Hidden and logits
     * policies are independent because MTP transactions may publish a hidden
     * mailbox while an ordinary partial-prefix suffix only recomputes logits.
     */
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

        /** Comparison contract used when main KV payload hashing is enabled. */
        MTPMainKVPayloadComparisonPolicy main_kv_payload_policy =
            MTPMainKVPayloadComparisonPolicy::ExactBytes;

        /**
         * Number of leading logical tokens that must remain byte-identical.
         *
         * A non-negative value is mandatory for the placement-aware policy;
         * the remaining live tokens form the numerically compared suffix.
         */
        int main_kv_exact_prefix_tokens = -1;

        /** Minimum cosine for every recomputed K and V suffix payload. */
        double main_kv_suffix_min_cosine =
            kDefaultPlacementAwareMainKVSuffixMinimumCosine;

        bool compare_shifted_mtp_kv = true;
        MTPGDNStateComparisonPolicy gdn_state_policy =
            MTPGDNStateComparisonPolicy::ExactBytes;
        MTPTerminalPayloadComparisonPolicy terminal_hidden_policy =
            MTPTerminalPayloadComparisonPolicy::ExactBytes;
        /** Minimum full-row cosine after a proven MoE placement change. */
        double terminal_hidden_min_cosine =
            kDefaultPlacementAwareTerminalPayloadMinimumCosine;
        MTPTerminalPayloadComparisonPolicy terminal_logits_policy =
            MTPTerminalPayloadComparisonPolicy::ExactBytes;
        /** Minimum full-logits-row cosine after a proven placement change. */
        double terminal_logits_min_cosine =
            kDefaultPlacementAwareTerminalPayloadMinimumCosine;
        /** Optional scale-aware bound; nullopt records but does not gate it. */
        std::optional<double> gdn_relative_l2_tolerance = 1e-4;
        /** Optional scale-dependent bound; nullopt records but does not gate it. */
        std::optional<double> gdn_max_abs_tolerance = 1e-4;
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
