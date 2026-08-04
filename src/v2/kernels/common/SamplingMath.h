/**
 * @file SamplingMath.h
 * @brief Shared CPU/CUDA/ROCm stochastic sampling math.
 */
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "utils/PrefillGraphBucketDefaults.h"

#if defined(__CUDACC__) || defined(__HIPCC__)
#define LLAMINAR_SAMPLING_HD __host__ __device__ inline
#else
#define LLAMINAR_SAMPLING_HD inline
#endif

namespace llaminar2::sampling_math
{
    constexpr int kMaxTopK = 256;

    /**
     * @brief Default number of draft-token comparisons in one MTP transaction.
     *
     * A target verifier evaluates one row per draft token plus a terminal bonus
     * row.  The shared grouped-verifier capacity therefore leaves exactly one
     * row for that bonus.  Kernels consume the runtime row count and do not
     * specialize on this value; it sizes the default graph-owned mailboxes and
     * value-owned launch parameters only.
     */
    constexpr int kSpeculativeBatchMaxRows =
        kDefaultNativeVNNIVerifierRowCapacity - 1;
    constexpr int kSpeculativeBatchMaxOutputTokens =
        kSpeculativeBatchMaxRows + 1;
    constexpr int kSpeculativeBatchMaxStopTokens = 8;
    constexpr int kSpeculativeBatchMetaCount = 12;
    constexpr float kMaxUnitThreshold = 0.99999994f;
    constexpr uint64_t kInverseSampleDomain = 0xA0761D6478BD642FULL;
    constexpr uint64_t kMTPSpecDrawPurposesPerToken = 8;

    /**
     * @brief ABI version for the retained first-transaction MTP diagnostic.
     *
     * The record is copied byte-for-byte from device storage only after a
     * mirrored participant mismatch has already made inference fatal. Keeping
     * an explicit version in the payload prevents a stale host formatter from
     * silently interpreting a changed device layout.
     */
    constexpr uint32_t kMTPFirstTransactionDiagnosticVersion = 2;

    /**
     * @brief Ordered MTP-sidecar boundaries retained for transaction zero.
     *
     * The proposal graph reuses one activation workspace for every draft and
     * every device-controlled transaction.  A terminal failure therefore sees
     * only the last sidecar that happened to execute.  These stable boundary
     * identifiers let the captured draft-publication stage preserve a compact
     * digest immediately after each sidecar, before the shared workspace is
     * overwritten by the next proposal.
     */
    enum class MTPFirstTransactionDraftBoundary : int32_t
    {
        TerminalHiddenInput = 0,
        Embedding,
        NormalizedTerminalHidden,
        NormalizedEmbedding,
        ConcatenatedInput,
        ProjectedInput,
        AttentionQuery,
        AttentionKey,
        AttentionValue,
        AttentionOutput,
        AttentionProjection,
        MoEExpertIndices,
        MoEExpertWeights,
        MoERoutedOutput,
        MoESharedOutput,
        FFNOutput,
        FinalHidden,
        ScoredLogits,
        Count,
    };
    constexpr int kMTPFirstTransactionDraftBoundaryCount =
        static_cast<int>(MTPFirstTransactionDraftBoundary::Count);

    /**
     * @brief Device-resident evidence from the first stochastic MTP transaction.
     *
     * A device-controlled generation graph may execute several verifier
     * transactions before its single terminal host observation. Ordinary
     * scratch buffers therefore contain the final transaction, which is too
     * late to diagnose a participant divergence that happened in transaction
     * zero. The fused serial-equivalent outcome kernel optionally records this
     * fixed-size payload before publication advances the device controller.
     *
     * The payload deliberately retains hashes rather than complete Top-K rows.
     * Each hash covers the exact token-id and FP32 probability bits in canonical
     * scan order. Together with the seed, logical position, thresholds, sampled
     * tokens, verifier inputs, and compact result, this distinguishes four
     * boundaries without perturbing the production graph topology:
     *
     *  - unequal target-distribution hashes identify grouped-forward drift;
     *  - equal hashes with unequal thresholds identify RNG-position drift;
     *  - equal hashes/thresholds with unequal samples identify kernel drift;
     *  - equal samples with unequal compact results identify reducer drift.
     *
     * No pointer is stored in the record. It is a trivially copyable shared ABI
     * consumed by CUDA, ROCm, and exceptional-path host diagnostics.
     */
    struct alignas(8) MTPFirstTransactionDiagnosticRecord
    {
        uint32_t valid = 0; ///< Written last by lane zero after the record is complete.
        uint32_t version = kMTPFirstTransactionDiagnosticVersion;
        uint64_t threshold_seed = 0;
        int32_t threshold_base_position = -1;
        int32_t threshold_position_offset = 0;
        int32_t comparison_row_count = 0;
        int32_t top_k = 0;
        int32_t transaction_count = -1;
        int32_t transaction_commit_budget = 0;
        int32_t leading_committed_output_count = 0;
        int32_t verifier_input_tokens[kSpeculativeBatchMaxOutputTokens] = {};
        int32_t sampled_target_tokens[kSpeculativeBatchMaxOutputTokens] = {};
        int32_t sampled_matches_verifier_input[kSpeculativeBatchMaxOutputTokens] = {};
        uint32_t threshold_bits[kSpeculativeBatchMaxOutputTokens] = {};
        uint64_t target_distribution_hashes[kSpeculativeBatchMaxOutputTokens] = {};
        int32_t output_tokens[kSpeculativeBatchMaxOutputTokens] = {};
        int32_t output_meta[kSpeculativeBatchMetaCount] = {};

        /*
         * Proposal-side evidence is written by the captured draft publication
         * fragments before the verifier runs.  Each digest covers the exact
         * 32-bit words of one current sidecar row.  The word count is retained
         * beside the hash so absent model-specific boundaries cannot compare
         * equal to a real zero-filled tensor by accident.
         */
        int32_t draft_diagnostic_depth = 0;
        int32_t draft_condition_tokens[kSpeculativeBatchMaxRows] = {};
        int32_t draft_position_ids[kSpeculativeBatchMaxRows] = {};
        uint32_t
            draft_boundary_word_counts[kSpeculativeBatchMaxRows]
                                      [kMTPFirstTransactionDraftBoundaryCount] = {};
        uint64_t
            draft_boundary_hashes[kSpeculativeBatchMaxRows]
                                  [kMTPFirstTransactionDraftBoundaryCount] = {};
    };
    static_assert(
        sizeof(MTPFirstTransactionDiagnosticRecord) % sizeof(int32_t) == 0,
        "The arena stores the first-transaction diagnostic as whole INT32 words");

    /**
     * @brief Return the exact IEEE-754 bit pattern used by sampling math.
     *
     * CUDA and HIP both support this scalar union representation in device
     * code. Recording bits rather than formatting floats on-device preserves
     * byte equality and leaves presentation to the fatal host diagnostic.
     */
    LLAMINAR_SAMPLING_HD uint32_t sampling_float_bits(float value)
    {
        union FloatBits
        {
            float fp32;
            uint32_t bits;
        } converted{};
        converted.fp32 = value;
        return converted.bits;
    }

    /**
     * @brief Append one 32-bit word to a byte-ordered FNV-1a diagnostic hash.
     */
    LLAMINAR_SAMPLING_HD uint64_t append_diagnostic_u32_fnv1a(
        uint64_t hash,
        uint32_t value)
    {
        constexpr uint64_t kFNVPrime = 1099511628211ULL;
        for (unsigned int byte = 0; byte < sizeof(value); ++byte)
        {
            hash ^= static_cast<uint8_t>(value >> (byte * 8U));
            hash *= kFNVPrime;
        }
        return hash;
    }

    /**
     * @brief Hash one compact target row in its serial sampling scan order.
     */
    LLAMINAR_SAMPLING_HD uint64_t hash_compact_distribution_exact(
        const int32_t *token_ids,
        const float *probabilities,
        int top_k)
    {
        uint64_t hash = 1469598103934665603ULL;
        for (int column = 0; column < top_k; ++column)
        {
            hash = append_diagnostic_u32_fnv1a(
                hash,
                static_cast<uint32_t>(token_ids[column]));
            hash = append_diagnostic_u32_fnv1a(
                hash,
                sampling_float_bits(probabilities[column]));
        }
        return hash;
    }

    /**
     * @brief Populate one deterministic row of transaction-zero evidence.
     *
     * Every physical diagnostic row is initialized, including rows outside the
     * active depth. This prevents stale bytes from an earlier request from
     * creating a false mirrored mismatch when two participants previously ran
     * different verifier depths.
     */
    LLAMINAR_SAMPLING_HD void record_mtp_first_transaction_sample_row(
        MTPFirstTransactionDiagnosticRecord *record,
        int row,
        bool active,
        const int32_t *target_token_ids,
        const float *target_probabilities,
        int top_k,
        const int32_t *verifier_input_tokens,
        int comparison_row_count,
        float threshold,
        int32_t sampled_token)
    {
        if (!record || row < 0 ||
            row >= kSpeculativeBatchMaxOutputTokens)
        {
            return;
        }

        record->verifier_input_tokens[row] =
            active && verifier_input_tokens
                ? verifier_input_tokens[row]
                : -1;
        record->sampled_target_tokens[row] =
            active ? sampled_token : -1;
        record->sampled_matches_verifier_input[row] =
            active && row < comparison_row_count && verifier_input_tokens
                ? (sampled_token == verifier_input_tokens[row + 1] ? 1 : 0)
                : -1;
        record->threshold_bits[row] =
            active ? sampling_float_bits(threshold) : 0u;
        record->target_distribution_hashes[row] =
            active && target_token_ids && target_probabilities
                ? hash_compact_distribution_exact(
                      target_token_ids,
                      target_probabilities,
                      top_k)
                : 0ULL;
    }

    /**
     * @brief Complete transaction-zero evidence after compact reduction.
     *
     * This function is called by lane zero only, after the workgroup barrier
     * and after the ordinary serial-equivalent reducer has populated its output
     * rows. `valid` is written last so an event-ordered observer can reject an
     * incomplete or ABI-incompatible record without guessing.
     */
    LLAMINAR_SAMPLING_HD void finalize_mtp_first_transaction_diagnostic(
        MTPFirstTransactionDiagnosticRecord *record,
        uint64_t threshold_seed,
        int threshold_base_position,
        int threshold_position_offset,
        int comparison_row_count,
        int top_k,
        int transaction_count,
        int transaction_commit_budget,
        int leading_committed_output_count,
        const int32_t *output_tokens,
        int output_token_capacity,
        const int32_t *output_meta)
    {
        if (!record)
            return;

        record->valid = 0;
        record->version = kMTPFirstTransactionDiagnosticVersion;
        record->threshold_seed = threshold_seed;
        record->threshold_base_position = threshold_base_position;
        record->threshold_position_offset = threshold_position_offset;
        record->comparison_row_count = comparison_row_count;
        record->top_k = top_k;
        record->transaction_count = transaction_count;
        record->transaction_commit_budget = transaction_commit_budget;
        record->leading_committed_output_count =
            leading_committed_output_count;

        for (int slot = 0;
             slot < kSpeculativeBatchMaxOutputTokens;
             ++slot)
        {
            record->output_tokens[slot] =
                output_tokens && slot < output_token_capacity
                    ? output_tokens[slot]
                    : -1;
        }
        for (int field = 0; field < kSpeculativeBatchMetaCount; ++field)
        {
            record->output_meta[field] =
                output_meta ? output_meta[field] : 0;
        }
        record->valid = 1;
    }

    /**
     * @brief Value-owned explicit thresholds captured with one GPU launch.
     *
     * Production seeded MTP derives these values from device-owned logical
     * positions, but explicit-threshold integration probes still need a stable
     * graph-capturable launch ABI. Passing one bounded aggregate avoids a
     * depth-specialized argument list and keeps every runtime row addressable.
     */
    struct SpeculativeBatchThresholdParameters
    {
        float accept[kSpeculativeBatchMaxRows] = {};
        float residual[kSpeculativeBatchMaxRows] = {};
    };

    /**
     * @brief Value-owned host-token inputs for the diagnostic batch verifier.
     *
     * The production GPU path reads sampled draft tokens from device buffers.
     * This aggregate exists for explicit low-level probes and contains no
     * pointers whose lifetime could outlive graph capture.
     */
    struct SpeculativeBatchHostParameters
    {
        int draft_tokens[kSpeculativeBatchMaxRows] = {};
        SpeculativeBatchThresholdParameters thresholds{};
    };

    enum SpeculativeBatchMetaIndex : int
    {
        kSpecBatchMetaOk = 0,
        kSpecBatchMetaOutputCount = 1,
        kSpecBatchMetaAcceptedSpeculativePrefix = 2,
        kSpecBatchMetaTargetVerifierStateCommitCount = 3,
        kSpecBatchMetaReadyToken = 4,
        kSpecBatchMetaRejectedVerifiedToken = 5,
        kSpecBatchMetaStoppedOnOutput = 6,
        kSpecBatchMetaAllSpeculativeAccepted = 7,
        kSpecBatchMetaConsumedVerifierRows = 8,
        kSpecBatchMetaSampledTerminal = 9,
        kSpecBatchMetaCommitBoundaryClipped = 10,
        /**
         * Number of leading compact output rows emitted by an earlier transaction.
         *
         * A rejected correction token is returned to the caller immediately, then
         * carried as row zero of the next verifier transaction so the model can
         * consume it as the next condition.  The row remains part of the compact
         * output/state-publication shape, but it must not advance a serial decode
         * cadence for a second time.
         */
        kSpecBatchMetaLeadingCommittedOutputCount = 11
    };

    /**
     * @brief Device-side MTP depth policy mode admitted before generation.
     */
    enum class DeviceGenerationDepthPolicyMode : int
    {
        Fixed = 0,
        Observe = 1,
        Dynamic = 2,
    };

    /**
     * @brief Fixed-width request policy consumed by the resident depth controller.
     *
     * Floating-point policy thresholds are converted to parts-per-million by
     * request admission. Device transitions can consequently compare integer
     * counters with a fixed arithmetic order on CUDA and ROCm, and CPU tests can
     * prove the same decisions without backend-specific floating-point drift.
     */
    struct DeviceGenerationDepthPolicy
    {
        static constexpr int kRateScale = 1'000'000;
        static constexpr int kMaximumSupportedDraftDepth = 15;

        DeviceGenerationDepthPolicyMode mode =
            DeviceGenerationDepthPolicyMode::Fixed;
        int initial_depth = 1;
        int minimum_depth = 1;
        int maximum_depth = 1;
        int window_size = 16;
        int minimum_samples = 4;
        int cooldown_steps = 8;
        int promote_consecutive_windows = 3;
        int promote_full_accept_rate_ppm = kRateScale;
        int demote_zero_accept_rate_ppm = 300'000;
        int demote_acceptance_rate_ppm = 550'000;

        /**
         * @brief Construct a hard-pinned policy for tests and fixed-depth lanes.
         */
        LLAMINAR_SAMPLING_HD static DeviceGenerationDepthPolicy fixed(
            int depth)
        {
            DeviceGenerationDepthPolicy policy;
            policy.mode = DeviceGenerationDepthPolicyMode::Fixed;
            policy.initial_depth = depth;
            policy.minimum_depth = depth;
            policy.maximum_depth = depth;
            return policy;
        }

        /**
         * @brief Validate all fields without consulting host configuration.
         */
        LLAMINAR_SAMPLING_HD bool valid() const
        {
            const bool mode_valid =
                mode == DeviceGenerationDepthPolicyMode::Fixed ||
                mode == DeviceGenerationDepthPolicyMode::Observe ||
                mode == DeviceGenerationDepthPolicyMode::Dynamic;
            const bool rates_valid =
                promote_full_accept_rate_ppm >= 0 &&
                promote_full_accept_rate_ppm <= kRateScale &&
                demote_zero_accept_rate_ppm >= 0 &&
                demote_zero_accept_rate_ppm <= kRateScale &&
                demote_acceptance_rate_ppm >= 0 &&
                demote_acceptance_rate_ppm <= kRateScale;
            return mode_valid && minimum_depth > 0 &&
                   maximum_depth >= minimum_depth &&
                   maximum_depth <= kMaximumSupportedDraftDepth &&
                   initial_depth >= minimum_depth &&
                   initial_depth <= maximum_depth && window_size > 0 &&
                   minimum_samples > 0 && cooldown_steps >= 0 &&
                   promote_consecutive_windows > 0 && rates_valid &&
                   (mode != DeviceGenerationDepthPolicyMode::Fixed ||
                    (minimum_depth == maximum_depth &&
                     initial_depth == minimum_depth));
        }
    };

    /**
     * @brief Stable control-word layout for a device-owned generation request.
     *
     * A speculative verifier transaction produces a compact token row and a
     * metadata row.  Production GPU generation must be able to consume many of
     * those transactions without asking the host how many response tokens have
     * already been emitted, whether a rejected correction row is being carried
     * into the next verifier, or whether the request has reached its terminal
     * budget.  These words are the single device-resident authority for those
     * facts.
     *
     * The layout intentionally contains only fixed-width INT32 values.  CUDA and
     * ROCm kernels can therefore share the exact transition helper below, while
     * CPU unit tests can exhaustively prove the same state machine without a GPU.
     * The response-token payload lives in a separate persistent arena row.
     */
    enum DeviceGenerationControlIndex : int
    {
        kDeviceGenerationControlOk = 0,
        kDeviceGenerationControlResponseTokenCount = 1,
        kDeviceGenerationControlRemainingTokenCount = 2,
        kDeviceGenerationControlModelStopped = 3,
        kDeviceGenerationControlRequestComplete = 4,
        kDeviceGenerationControlTransactionCount = 5,
        kDeviceGenerationControlNextLeadingCommittedOutputCount = 6,
        kDeviceGenerationControlTransactionCommitBudget = 7,
        kDeviceGenerationControlAcceptedSpeculativeTokenCount = 8,
        kDeviceGenerationControlRejectedTransactionCount = 9,
        kDeviceGenerationControlConsumedVerifierRowCount = 10,
        kDeviceGenerationControlErrorCode = 11,
        /** Total main-graph state rows committed across every transaction. */
        kDeviceGenerationControlPublishedStateCommitCount = 12,
        /** Active SWITCH selector and number of speculative comparison rows. */
        kDeviceGenerationControlCurrentDraftDepth = 13,
        /** Logical verifier width, including the leading condition row. */
        kDeviceGenerationControlActiveVerifierRowCount = 14,
        kDeviceGenerationControlMinimumDraftDepth = 15,
        kDeviceGenerationControlMaximumDraftDepth = 16,
        kDeviceGenerationControlDepthPolicyMode = 17,
        kDeviceGenerationControlDepthWindowSize = 18,
        kDeviceGenerationControlDepthMinimumSamples = 19,
        kDeviceGenerationControlDepthCooldownSteps = 20,
        kDeviceGenerationControlDepthPromoteConsecutiveWindows = 21,
        kDeviceGenerationControlDepthPromoteFullAcceptRatePPM = 22,
        kDeviceGenerationControlDepthDemoteZeroAcceptRatePPM = 23,
        kDeviceGenerationControlDepthDemoteAcceptanceRatePPM = 24,
        kDeviceGenerationControlDepthStepsSinceChange = 25,
        kDeviceGenerationControlDepthPromotionStreak = 26,
        kDeviceGenerationControlDepthWindowVerifierRuns = 27,
        kDeviceGenerationControlDepthWindowAttemptedTokens = 28,
        kDeviceGenerationControlDepthWindowAcceptedTokens = 29,
        kDeviceGenerationControlDepthWindowRejectedTokens = 30,
        kDeviceGenerationControlDepthWindowRollbacks = 31,
        kDeviceGenerationControlDepthWindowFullAccepts = 32,
        kDeviceGenerationControlDepthWindowZeroAccepts = 33,
        kDeviceGenerationControlDepthWindowAcceptedPrefixSum = 34,
        kDeviceGenerationControlDepthEvaluatedWindows = 35,
        kDeviceGenerationControlDepthUpdates = 36,
        kDeviceGenerationControlDepthPromotions = 37,
        kDeviceGenerationControlDepthDemotions = 38,
        kDeviceGenerationControlDepthLastRecommendedDepth = 39,
        /** Sum of device-selected speculative widths across transactions. */
        kDeviceGenerationControlAttemptedDraftTokenCount = 40,
        /** Sum of logical verifier widths, including each condition row. */
        kDeviceGenerationControlVerifierTokenCount = 41,
        kDeviceGenerationControlCount = 42,
    };

    /**
     * @brief Fatal validation failures reported by the generation controller.
     *
     * The controller never repairs or truncates malformed production output.
     * A non-zero value invalidates the request and is surfaced by the single
     * terminal materialization.  This prevents a GPU request from limping on
     * after a stale mailbox, response-budget overrun, or compact-ABI drift.
     */
    enum class DeviceGenerationError : int
    {
        None = 0,
        InvalidInitialization = 1,
        InvalidController = 2,
        InvalidCompactOutcome = 3,
        LeadingCommittedRowMismatch = 4,
        EmptyTransaction = 5,
        ResponseBudgetExceeded = 6,
        ResponseCapacityExceeded = 7,
        InvalidVerifierCounts = 8,
        InvalidPublicationMetadata = 9,
        InvalidDepthPolicy = 10,
        InvalidDepthSelector = 11,
    };

    /**
     * @brief Invalidate a resident generation request after a fatal transition.
     *
     * Keeping failure publication in a named host/device helper avoids
     * compiler-specific device lambdas and gives CUDA, ROCm, and CPU tests one
     * exact fail-hard state.  A failed request is terminal: later graph replays
     * observe `Ok == 0`, publish no commit budget, and preserve the first error
     * code for terminal materialization.
     *
     * @param control Mutable controller row, or nullptr when pointer validation
     *        itself failed.
     * @param error Stable error code to surface at the request boundary.
     * @return Always false, allowing callers to return the transition result.
     */
    LLAMINAR_SAMPLING_HD bool fail_device_generation_control(
        int *control,
        DeviceGenerationError error)
    {
        if (control)
        {
            control[kDeviceGenerationControlOk] = 0;
            control[kDeviceGenerationControlRequestComplete] = 1;
            control[kDeviceGenerationControlTransactionCommitBudget] = 0;
            control[kDeviceGenerationControlErrorCode] =
                static_cast<int>(error);
        }
        return false;
    }

    /**
     * @brief Initialize one persistent device generation controller row.
     *
     * This helper is called by a tiny explicit-stream backend kernel at request
     * admission.  The host supplies immutable request policy exactly once; all
     * later transaction counts and boundaries are device-owned.
     *
     * @param max_new_tokens Number of response tokens requested by the caller.
     * @param response_capacity Number of token slots in the persistent response
     *        row.  It must cover the complete request budget.
     * @param depth_policy Immutable fixed/dynamic policy admitted for this request.
     * @param control Writable row with @ref kDeviceGenerationControlCount words.
     * @return true when the initialized controller is valid.
     */
    LLAMINAR_SAMPLING_HD bool initialize_device_generation_control(
        int max_new_tokens,
        int response_capacity,
        const DeviceGenerationDepthPolicy &depth_policy,
        int *control)
    {
        if (!control)
            return false;

        for (int i = 0; i < kDeviceGenerationControlCount; ++i)
            control[i] = 0;

        if (max_new_tokens <= 0 ||
            response_capacity <= 0 ||
            max_new_tokens > response_capacity)
        {
            control[kDeviceGenerationControlErrorCode] =
                static_cast<int>(DeviceGenerationError::InvalidInitialization);
            return false;
        }
        if (!depth_policy.valid())
        {
            control[kDeviceGenerationControlErrorCode] =
                static_cast<int>(DeviceGenerationError::InvalidDepthPolicy);
            return false;
        }

        control[kDeviceGenerationControlOk] = 1;
        control[kDeviceGenerationControlRemainingTokenCount] = max_new_tokens;
        control[kDeviceGenerationControlCurrentDraftDepth] =
            depth_policy.initial_depth;
        control[kDeviceGenerationControlActiveVerifierRowCount] =
            depth_policy.initial_depth + 1;
        control[kDeviceGenerationControlMinimumDraftDepth] =
            depth_policy.minimum_depth;
        control[kDeviceGenerationControlMaximumDraftDepth] =
            depth_policy.maximum_depth;
        control[kDeviceGenerationControlDepthPolicyMode] =
            static_cast<int>(depth_policy.mode);
        control[kDeviceGenerationControlDepthWindowSize] =
            depth_policy.window_size;
        control[kDeviceGenerationControlDepthMinimumSamples] =
            depth_policy.minimum_samples;
        control[kDeviceGenerationControlDepthCooldownSteps] =
            depth_policy.cooldown_steps;
        control[kDeviceGenerationControlDepthPromoteConsecutiveWindows] =
            depth_policy.promote_consecutive_windows;
        control[kDeviceGenerationControlDepthPromoteFullAcceptRatePPM] =
            depth_policy.promote_full_accept_rate_ppm;
        control[kDeviceGenerationControlDepthDemoteZeroAcceptRatePPM] =
            depth_policy.demote_zero_accept_rate_ppm;
        control[kDeviceGenerationControlDepthDemoteAcceptanceRatePPM] =
            depth_policy.demote_acceptance_rate_ppm;
        control[kDeviceGenerationControlDepthStepsSinceChange] =
            depth_policy.cooldown_steps;
        control[kDeviceGenerationControlDepthLastRecommendedDepth] =
            depth_policy.initial_depth;
        return true;
    }

    /**
     * @brief Publish the maximum serial-visible row count for the next verifier.
     *
     * The verifier reducer already accepts a device pointer for its commit
     * boundary.  This transition combines the response budget with the current
     * device-owned maintenance boundary before any verifier output is reduced.
     * A completed request publishes zero, allowing a graph-owned conditional or
     * no-op admission kernel to suppress later work without a host poll.
     *
     * @param verifier_row_capacity Static verifier graph row capacity.
     * @param maintenance_rows_remaining Device MoE maintenance boundary.  Values
     *        <= 0 are invalid because maintenance must run before another
     *        serial-visible transaction is admitted.
     * @param control Mutable generation controller row.
     * @return Published commit budget, or zero when the request is complete or
     *         invalid.
     */
    LLAMINAR_SAMPLING_HD int prepare_device_generation_transaction_budget(
        int verifier_row_capacity,
        int maintenance_rows_remaining,
        int *control)
    {
        if (!control || verifier_row_capacity <= 0 ||
            maintenance_rows_remaining <= 0)
        {
            if (control)
            {
                control[kDeviceGenerationControlOk] = 0;
                control[kDeviceGenerationControlErrorCode] =
                    static_cast<int>(DeviceGenerationError::InvalidController);
                control[kDeviceGenerationControlTransactionCommitBudget] = 0;
            }
            return 0;
        }
        if (control[kDeviceGenerationControlOk] == 0 ||
            control[kDeviceGenerationControlRequestComplete] != 0)
        {
            control[kDeviceGenerationControlTransactionCommitBudget] = 0;
            return 0;
        }

        const int remaining =
            control[kDeviceGenerationControlRemainingTokenCount];
        if (remaining <= 0)
        {
            control[kDeviceGenerationControlRequestComplete] = 1;
            control[kDeviceGenerationControlTransactionCommitBudget] = 0;
            return 0;
        }

        const int active_verifier_rows =
            control[kDeviceGenerationControlActiveVerifierRowCount];
        if (active_verifier_rows <= 1 ||
            active_verifier_rows > verifier_row_capacity)
        {
            fail_device_generation_control(
                control,
                DeviceGenerationError::InvalidDepthSelector);
            return 0;
        }

        int budget = remaining < active_verifier_rows
                         ? remaining
                         : active_verifier_rows;
        budget = budget < maintenance_rows_remaining
                     ? budget
                     : maintenance_rows_remaining;
        control[kDeviceGenerationControlTransactionCommitBudget] = budget;
        return budget;
    }

    /**
     * @brief Compare one integer ratio with a PPM threshold without division.
     */
    LLAMINAR_SAMPLING_HD bool device_generation_rate_at_least(
        int numerator,
        int denominator,
        int threshold_ppm)
    {
        if (numerator < 0 || denominator <= 0 ||
            threshold_ppm < 0 ||
            threshold_ppm > DeviceGenerationDepthPolicy::kRateScale)
        {
            return false;
        }
        return static_cast<int64_t>(numerator) *
                   DeviceGenerationDepthPolicy::kRateScale >=
               static_cast<int64_t>(denominator) * threshold_ppm;
    }

    /**
     * @brief Record one verifier outcome and publish the next device depth.
     *
     * The transition intentionally uses only integer counters. Fixed mode does
     * no adaptive bookkeeping. Observe mode evaluates the same windows and
     * records `LastRecommendedDepth` while retaining the active selector.
     * Dynamic mode applies one-step promotion/demotion with cooldown and
     * promotion hysteresis. The complete policy state remains in this row and
     * is consumed by the next native SWITCH iteration without host polling.
     */
    LLAMINAR_SAMPLING_HD bool record_device_generation_depth_observation(
        int accepted_prefix,
        bool rollback,
        bool budget_limited,
        int *control)
    {
        if (!control || control[kDeviceGenerationControlOk] == 0)
            return false;

        const int mode = control[kDeviceGenerationControlDepthPolicyMode];
        const int current =
            control[kDeviceGenerationControlCurrentDraftDepth];
        const int minimum =
            control[kDeviceGenerationControlMinimumDraftDepth];
        const int maximum =
            control[kDeviceGenerationControlMaximumDraftDepth];
        if (current < minimum || current > maximum || minimum <= 0 ||
            maximum > DeviceGenerationDepthPolicy::kMaximumSupportedDraftDepth ||
            (mode != static_cast<int>(DeviceGenerationDepthPolicyMode::Fixed) &&
             mode != static_cast<int>(DeviceGenerationDepthPolicyMode::Observe) &&
             mode != static_cast<int>(DeviceGenerationDepthPolicyMode::Dynamic)))
        {
            return fail_device_generation_control(
                control,
                DeviceGenerationError::InvalidDepthPolicy);
        }
        if (mode == static_cast<int>(DeviceGenerationDepthPolicyMode::Fixed) ||
            budget_limited)
        {
            return true;
        }

        if (accepted_prefix < 0 || accepted_prefix > current)
        {
            return fail_device_generation_control(
                control,
                DeviceGenerationError::InvalidVerifierCounts);
        }

        ++control[kDeviceGenerationControlDepthWindowVerifierRuns];
        control[kDeviceGenerationControlDepthWindowAttemptedTokens] += current;
        control[kDeviceGenerationControlDepthWindowAcceptedTokens] +=
            accepted_prefix;
        control[kDeviceGenerationControlDepthWindowRejectedTokens] +=
            current - accepted_prefix;
        control[kDeviceGenerationControlDepthWindowAcceptedPrefixSum] +=
            accepted_prefix;
        control[kDeviceGenerationControlDepthWindowRollbacks] +=
            rollback ? 1 : 0;
        control[kDeviceGenerationControlDepthWindowFullAccepts] +=
            accepted_prefix == current ? 1 : 0;
        control[kDeviceGenerationControlDepthWindowZeroAccepts] +=
            accepted_prefix == 0 ? 1 : 0;
        ++control[kDeviceGenerationControlDepthStepsSinceChange];

        const int verifier_runs =
            control[kDeviceGenerationControlDepthWindowVerifierRuns];
        const int required_samples =
            control[kDeviceGenerationControlDepthWindowSize] >
                    control[kDeviceGenerationControlDepthMinimumSamples]
                ? control[kDeviceGenerationControlDepthWindowSize]
                : control[kDeviceGenerationControlDepthMinimumSamples];
        if (verifier_runs < required_samples)
            return true;

        int recommended = current;
        int promotion_streak =
            control[kDeviceGenerationControlDepthPromotionStreak];
        const bool cooldown_complete =
            control[kDeviceGenerationControlDepthStepsSinceChange] >=
            control[kDeviceGenerationControlDepthCooldownSteps];
        const int attempted =
            control[kDeviceGenerationControlDepthWindowAttemptedTokens];
        const int accepted =
            control[kDeviceGenerationControlDepthWindowAcceptedTokens];
        const int zero_accepts =
            control[kDeviceGenerationControlDepthWindowZeroAccepts];
        const int full_accepts =
            control[kDeviceGenerationControlDepthWindowFullAccepts];

        if (cooldown_complete && current > minimum &&
            device_generation_rate_at_least(
                zero_accepts,
                verifier_runs,
                control[
                    kDeviceGenerationControlDepthDemoteZeroAcceptRatePPM]))
        {
            recommended = current - 1;
            promotion_streak = 0;
        }
        else if (cooldown_complete && current > minimum &&
                 !device_generation_rate_at_least(
                     accepted,
                     attempted,
                     control[
                         kDeviceGenerationControlDepthDemoteAcceptanceRatePPM]))
        {
            recommended = current - 1;
            promotion_streak = 0;
        }
        else if (cooldown_complete && current < maximum &&
                 zero_accepts == 0 &&
                 device_generation_rate_at_least(
                     full_accepts,
                     verifier_runs,
                     control[
                         kDeviceGenerationControlDepthPromoteFullAcceptRatePPM]))
        {
            ++promotion_streak;
            if (promotion_streak >=
                control[
                    kDeviceGenerationControlDepthPromoteConsecutiveWindows])
            {
                recommended = current + 1;
                promotion_streak = 0;
            }
        }
        else
        {
            promotion_streak = 0;
        }

        control[kDeviceGenerationControlDepthPromotionStreak] =
            promotion_streak;
        control[kDeviceGenerationControlDepthLastRecommendedDepth] =
            recommended;
        ++control[kDeviceGenerationControlDepthEvaluatedWindows];

        if (mode == static_cast<int>(DeviceGenerationDepthPolicyMode::Dynamic) &&
            recommended != current)
        {
            control[kDeviceGenerationControlCurrentDraftDepth] = recommended;
            control[kDeviceGenerationControlActiveVerifierRowCount] =
                recommended + 1;
            control[kDeviceGenerationControlDepthStepsSinceChange] = 0;
            ++control[kDeviceGenerationControlDepthUpdates];
            if (recommended > current)
                ++control[kDeviceGenerationControlDepthPromotions];
            else
                ++control[kDeviceGenerationControlDepthDemotions];
        }

        control[kDeviceGenerationControlDepthWindowVerifierRuns] = 0;
        control[kDeviceGenerationControlDepthWindowAttemptedTokens] = 0;
        control[kDeviceGenerationControlDepthWindowAcceptedTokens] = 0;
        control[kDeviceGenerationControlDepthWindowRejectedTokens] = 0;
        control[kDeviceGenerationControlDepthWindowRollbacks] = 0;
        control[kDeviceGenerationControlDepthWindowFullAccepts] = 0;
        control[kDeviceGenerationControlDepthWindowZeroAccepts] = 0;
        control[kDeviceGenerationControlDepthWindowAcceptedPrefixSum] = 0;
        return true;
    }

    /**
     * @brief Append one compact verifier outcome to a device response ledger.
     *
     * The function enforces serial decode response semantics.  A rejected token
     * may be emitted at the end of transaction N and carried as verifier input
     * row zero in transaction N+1; the compact ABI marks that row with
     * `kSpecBatchMetaLeadingCommittedOutputCount`, and this controller refuses to
     * count it twice.  No malformed or over-budget output is silently clipped.
     *
     * @param compact_tokens Compact output-token row for one request.
     * @param output_token_stride Capacity of @p compact_tokens.
     * @param compact_meta Compact metadata row for the same request.
     * @param meta_stride Capacity of @p compact_meta.
     * @param response_tokens Persistent device response row.
     * @param response_capacity Capacity of @p response_tokens.
     * @param control Mutable controller row.
     * @return true when the transaction was appended, or when the request was
     *         already complete and therefore required no mutation.
     */
    LLAMINAR_SAMPLING_HD bool append_speculative_outcome_to_device_generation(
        const int32_t *compact_tokens,
        int output_token_stride,
        const int *compact_meta,
        int meta_stride,
        int32_t *response_tokens,
        int response_capacity,
        int *control)
    {
        if (!control || !compact_tokens || !compact_meta || !response_tokens ||
            output_token_stride <= 0 ||
            meta_stride < kSpeculativeBatchMetaCount ||
            response_capacity <= 0)
        {
            return fail_device_generation_control(
                control,
                DeviceGenerationError::InvalidController);
        }
        if (control[kDeviceGenerationControlOk] == 0)
            return false;
        if (control[kDeviceGenerationControlRequestComplete] != 0)
            return true;
        if (compact_meta[kSpecBatchMetaOk] == 0)
            return fail_device_generation_control(
                control,
                DeviceGenerationError::InvalidCompactOutcome);

        const int output_count = compact_meta[kSpecBatchMetaOutputCount];
        const int leading_count =
            compact_meta[kSpecBatchMetaLeadingCommittedOutputCount];
        const int expected_leading_count =
            control[kDeviceGenerationControlNextLeadingCommittedOutputCount];
        const int verifier_state_count =
            compact_meta[kSpecBatchMetaTargetVerifierStateCommitCount];
        const int accepted_prefix =
            compact_meta[kSpecBatchMetaAcceptedSpeculativePrefix];
        const int consumed_rows =
            compact_meta[kSpecBatchMetaConsumedVerifierRows];

        if (output_count <= 0 || output_count > output_token_stride ||
            leading_count < 0 || leading_count > 1 ||
            leading_count > output_count)
        {
            return fail_device_generation_control(
                control,
                DeviceGenerationError::InvalidCompactOutcome);
        }
        if (leading_count != expected_leading_count)
        {
            return fail_device_generation_control(
                control,
                DeviceGenerationError::LeadingCommittedRowMismatch);
        }
        if (verifier_state_count < 0 || verifier_state_count > output_count ||
            accepted_prefix < 0 || consumed_rows < 0)
        {
            return fail_device_generation_control(
                control,
                DeviceGenerationError::InvalidVerifierCounts);
        }

        const int newly_emitted_count = output_count - leading_count;
        if (newly_emitted_count <= 0)
        {
            return fail_device_generation_control(
                control,
                DeviceGenerationError::EmptyTransaction);
        }

        const int response_count =
            control[kDeviceGenerationControlResponseTokenCount];
        const int remaining_count =
            control[kDeviceGenerationControlRemainingTokenCount];
        if (response_count < 0 || remaining_count < newly_emitted_count)
        {
            return fail_device_generation_control(
                control,
                DeviceGenerationError::ResponseBudgetExceeded);
        }
        if (response_count + newly_emitted_count > response_capacity)
        {
            return fail_device_generation_control(
                control,
                DeviceGenerationError::ResponseCapacityExceeded);
        }

        for (int i = 0; i < newly_emitted_count; ++i)
        {
            response_tokens[response_count + i] =
                compact_tokens[leading_count + i];
        }

        const int next_response_count = response_count + newly_emitted_count;
        const int next_remaining_count = remaining_count - newly_emitted_count;
        const bool model_stopped =
            compact_meta[kSpecBatchMetaStoppedOnOutput] != 0;
        const int transaction_budget =
            control[kDeviceGenerationControlTransactionCommitBudget];
        /*
         * The compact outcome describes every physically evaluated verifier
         * row, while the transaction budget describes the serial-visible
         * prefix that may actually become live state.  Those counts differ at
         * a maintenance boundary.  In particular, a transaction entered with
         * a carried row zero may evaluate two physical rows while publishing
         * only row zero.  Its second output has already been emitted, but is
         * still the next transaction's condition token and must therefore be
         * skipped by that transaction's response append.
         *
         * Comparing output_count with verifier_state_count loses that carry:
         * verifier_state_count includes the row beyond the publication
         * boundary.  Derive the exact publication count from the same budget
         * consumed by derive_speculative_publication_metadata() so response,
         * KV, and recurrent-state ownership advance as one transaction.
         */
        const int published_state_count =
            verifier_state_count < transaction_budget
                ? verifier_state_count
                : transaction_budget;
        const bool has_emitted_pending_condition =
            !model_stopped && output_count > published_state_count;
        const bool rejected_transaction =
            !model_stopped &&
            compact_meta[kSpecBatchMetaAllSpeculativeAccepted] == 0 &&
            compact_meta[kSpecBatchMetaCommitBoundaryClipped] == 0;
        const int active_depth =
            control[kDeviceGenerationControlCurrentDraftDepth];
        const bool budget_limited =
            transaction_budget < active_depth + 1;

        control[kDeviceGenerationControlResponseTokenCount] =
            next_response_count;
        control[kDeviceGenerationControlRemainingTokenCount] =
            next_remaining_count;
        control[kDeviceGenerationControlModelStopped] = model_stopped ? 1 : 0;
        control[kDeviceGenerationControlRequestComplete] =
            model_stopped || next_remaining_count == 0 ? 1 : 0;
        control[kDeviceGenerationControlTransactionCount] += 1;
        control[kDeviceGenerationControlNextLeadingCommittedOutputCount] =
            has_emitted_pending_condition ? 1 : 0;
        control[kDeviceGenerationControlTransactionCommitBudget] = 0;
        control[kDeviceGenerationControlAcceptedSpeculativeTokenCount] +=
            accepted_prefix;
        control[kDeviceGenerationControlRejectedTransactionCount] +=
            rejected_transaction ? 1 : 0;
        control[kDeviceGenerationControlConsumedVerifierRowCount] +=
            consumed_rows;
        control[kDeviceGenerationControlPublishedStateCommitCount] +=
            published_state_count;
        control[kDeviceGenerationControlAttemptedDraftTokenCount] +=
            active_depth;
        control[kDeviceGenerationControlVerifierTokenCount] +=
            active_depth + 1;
        control[kDeviceGenerationControlErrorCode] =
            static_cast<int>(DeviceGenerationError::None);
        return record_device_generation_depth_observation(
            accepted_prefix,
            rejected_transaction,
            budget_limited,
            control);
    }

    /**
     * @brief Convert compact output rows into newly emitted serial decode rounds.
     *
     * The compact ABI permits exactly one leading row from an earlier transaction:
     * the pending rejection-correction condition.  Returning `-1` makes malformed
     * metadata fatal to device maintenance instead of silently double-counting it.
     */
    LLAMINAR_SAMPLING_HD int speculative_new_commit_count(
        int output_count,
        int leading_committed_output_count)
    {
        if (output_count <= 0 ||
            leading_committed_output_count < 0 ||
            leading_committed_output_count > 1 ||
            leading_committed_output_count > output_count)
        {
            return -1;
        }
        return output_count - leading_committed_output_count;
    }

    LLAMINAR_SAMPLING_HD uint64_t splitmix64(uint64_t x)
    {
        x += 0x9E3779B97F4A7C15ull;
        x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
        x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
        return x ^ (x >> 31);
    }

    LLAMINAR_SAMPLING_HD float uniform01(uint64_t seed, uint64_t offset)
    {
        const uint64_t bits = splitmix64(seed + offset);
        return static_cast<float>((bits >> 40) & 0xFFFFFFull) *
               (1.0f / 16777216.0f);
    }

    /**
     * @brief Deterministic MTP stochastic draw keyed by token position.
     *
     * vLLM-style speculative decoding may consume a draw in different graph
     * shapes: a bonus-ready row in one step can become the first token in the
     * next step, and verifier rows may be reduced entirely on device.  The draw
     * must therefore be keyed by logical output position and purpose instead
     * of by the host call order.  Keeping this helper shared lets CPU tests,
     * CUDA kernels, and ROCm kernels prove the same seeded thresholds.
     */
    LLAMINAR_SAMPLING_HD float mtp_spec_threshold_from_seed(
        uint64_t seed,
        int logical_position,
        int draw_purpose)
    {
        const uint64_t position =
            static_cast<uint64_t>(logical_position > 0 ? logical_position : 0);
        const uint64_t purpose =
            static_cast<uint64_t>(draw_purpose >= 0 ? draw_purpose : 0);
        return uniform01(
            seed,
            position * kMTPSpecDrawPurposesPerToken + purpose);
    }

    LLAMINAR_SAMPLING_HD float clamp_unit_threshold(float threshold)
    {
        return fminf(fmaxf(threshold, 0.0f), kMaxUnitThreshold);
    }

    /**
     * @brief Convert a uniform draw into vLLM-style inverse exponential noise.
     *
     * vLLM's stochastic rejection sampler picks recovered tokens by maximizing
     * `probability * inv_q[token]`, where `inv_q` is the reciprocal of an
     * exponential random variable. Keeping this tiny transform shared prevents
     * CPU, CUDA, and ROCm from drifting on clamp behavior near zero.
     */
    LLAMINAR_SAMPLING_HD float inverse_exponential_from_uniform(float uniform)
    {
        constexpr float kMinUniform = 1.0f / 16777216.0f;
        constexpr float kMinExponential = 1.0e-20f;
        const float u = fminf(fmaxf(uniform, kMinUniform), kMaxUnitThreshold);
        const float exponential = fmaxf(-logf(u), kMinExponential);
        return 1.0f / exponential;
    }

    LLAMINAR_SAMPLING_HD float speculative_accept_probability(
        float target_probability,
        float draft_probability)
    {
        if (!(target_probability > 0.0f) || !(draft_probability > 0.0f))
            return 0.0f;
        return fminf(1.0f, target_probability / draft_probability);
    }

    LLAMINAR_SAMPLING_HD float distribution_probability(
        const int *token_ids,
        const float *probs,
        int k,
        int token_id)
    {
        for (int i = 0; i < k; ++i)
        {
            if (token_ids[i] == token_id)
                return probs[i];
        }
        return 0.0f;
    }

    /**
     * @brief Sample a compact probability table and optionally return p(token).
     *
     * The selected probability is the stored compact-table probability for the
     * token that won the threshold scan. Keeping this helper shared prevents
     * CUDA, ROCm, and CPU-side verifier plumbing from drifting on edge cases
     * such as inactive token slots or clamped thresholds.
     */
    LLAMINAR_SAMPLING_HD int sample_distribution_with_threshold_and_probability(
        const int *token_ids,
        const float *probs,
        int k,
        float threshold,
        float *out_probability)
    {
        if (out_probability)
            *out_probability = 0.0f;

        float total = 0.0f;
        for (int i = 0; i < k; ++i)
        {
            if (token_ids[i] >= 0 && probs[i] > 0.0f)
                total += probs[i];
        }
        if (!(total > 0.0f))
            return -1;

        const float r = clamp_unit_threshold(threshold) * total;
        float cumulative = 0.0f;
        int selected = -1;
        for (int i = 0; i < k; ++i)
        {
            if (token_ids[i] < 0 || !(probs[i] > 0.0f))
                continue;
            if (selected < 0)
                selected = token_ids[i];
            cumulative += probs[i];
            if (r <= cumulative)
            {
                selected = token_ids[i];
                if (out_probability)
                    *out_probability = probs[i];
                break;
            }
        }
        if (out_probability && selected >= 0 && *out_probability == 0.0f)
            *out_probability = distribution_probability(token_ids, probs, k, selected);
        return selected;
    }

    LLAMINAR_SAMPLING_HD int build_topk_topp_distribution_from_sorted(
        const float *sorted_logits,
        const int *sorted_token_ids,
        int k,
        float top_p,
        float temperature,
        int *out_token_ids,
        float *out_probs,
        float *scratch_weights)
    {
        const float temp = temperature > 0.0f ? temperature : 1.0f;
        const float max_logit = sorted_logits[0];
        float total = 0.0f;
        for (int i = 0; i < k; ++i)
        {
            if (sorted_token_ids[i] < 0)
            {
                scratch_weights[i] = 0.0f;
                continue;
            }
            const float w = expf((sorted_logits[i] - max_logit) / temp);
            scratch_weights[i] = w;
            total += w;
        }

        int nucleus = k;
        if (total > 0.0f && top_p > 0.0f && top_p < 1.0f)
        {
            float cumulative = 0.0f;
            for (int i = 0; i < k; ++i)
            {
                cumulative += scratch_weights[i] / total;
                if (cumulative >= top_p)
                {
                    nucleus = i + 1;
                    break;
                }
            }
        }

        float nucleus_total = 0.0f;
        for (int i = 0; i < nucleus; ++i)
            nucleus_total += scratch_weights[i];

        int active = 0;
        for (int i = 0; i < k; ++i)
        {
            if (i < nucleus && nucleus_total > 0.0f && sorted_token_ids[i] >= 0)
            {
                out_token_ids[i] = sorted_token_ids[i];
                out_probs[i] = scratch_weights[i] / nucleus_total;
                ++active;
            }
            else
            {
                out_token_ids[i] = -1;
                out_probs[i] = 0.0f;
            }
        }
        return active;
    }

    LLAMINAR_SAMPLING_HD int sample_topk_topp_from_sorted_with_threshold(
        const float *sorted_logits,
        const int *sorted_token_ids,
        int k,
        float top_p,
        float temperature,
        float threshold,
        float *scratch_weights)
    {
        const float temp = temperature > 0.0f ? temperature : 1.0f;
        const float max_logit = sorted_logits[0];
        float total = 0.0f;
        for (int i = 0; i < k; ++i)
        {
            if (sorted_token_ids[i] < 0)
            {
                scratch_weights[i] = 0.0f;
                continue;
            }
            const float w = expf((sorted_logits[i] - max_logit) / temp);
            scratch_weights[i] = w;
            total += w;
        }

        if (!(total > 0.0f))
            return sorted_token_ids[0] >= 0 ? sorted_token_ids[0] : 0;

        int nucleus = k;
        if (top_p > 0.0f && top_p < 1.0f)
        {
            float cumulative = 0.0f;
            for (int i = 0; i < k; ++i)
            {
                cumulative += scratch_weights[i] / total;
                if (cumulative >= top_p)
                {
                    nucleus = i + 1;
                    break;
                }
            }
        }

        float nucleus_total = 0.0f;
        for (int i = 0; i < nucleus; ++i)
            nucleus_total += scratch_weights[i];
        if (!(nucleus_total > 0.0f))
            return sorted_token_ids[0] >= 0 ? sorted_token_ids[0] : 0;

        const float r = clamp_unit_threshold(threshold) * nucleus_total;
        float cumulative = 0.0f;
        int selected = sorted_token_ids[0] >= 0 ? sorted_token_ids[0] : 0;
        for (int i = 0; i < nucleus; ++i)
        {
            cumulative += scratch_weights[i];
            if (r <= cumulative)
            {
                selected = sorted_token_ids[i] >= 0 ? sorted_token_ids[i] : selected;
                break;
            }
        }
        return selected;
    }

    LLAMINAR_SAMPLING_HD int sample_distribution_with_threshold(
        const int *token_ids,
        const float *probs,
        int k,
        float threshold)
    {
        return sample_distribution_with_threshold_and_probability(
            token_ids,
            probs,
            k,
            threshold,
            nullptr);
    }

    LLAMINAR_SAMPLING_HD void speculative_verify_with_thresholds_and_draft_probability(
        const int *target_token_ids,
        const float *target_probs,
        const int *draft_token_ids,
        const float *draft_probs,
        int k,
        int draft_token,
        float sampled_draft_probability,
        bool has_sampled_draft_probability,
        float accept_threshold,
        float residual_threshold,
        int *out_token,
        int *out_accepted,
        float *out_accept_probability,
        float *out_accept_threshold)
    {
        const float p = distribution_probability(
            target_token_ids, target_probs, k, draft_token);
        const float q =
            has_sampled_draft_probability && sampled_draft_probability > 0.0f
                ? sampled_draft_probability
                : distribution_probability(draft_token_ids, draft_probs, k, draft_token);
        const float accept_probability = speculative_accept_probability(p, q);
        const float threshold = clamp_unit_threshold(accept_threshold);

        if (out_accept_probability)
            *out_accept_probability = accept_probability;
        if (out_accept_threshold)
            *out_accept_threshold = threshold;

        if (threshold < accept_probability)
        {
            *out_token = draft_token;
            *out_accepted = 1;
            return;
        }

        float residual_weights[kMaxTopK];
        float residual_total = 0.0f;
        for (int i = 0; i < k; ++i)
        {
            if (target_token_ids[i] < 0)
            {
                residual_weights[i] = 0.0f;
                continue;
            }
            const float q_i = distribution_probability(
                draft_token_ids,
                draft_probs,
                k,
                target_token_ids[i]);
            const float w = fmaxf(0.0f, target_probs[i] - q_i);
            residual_weights[i] = w;
            residual_total += w;
        }

        if (!(residual_total > 0.0f))
        {
            residual_total = 0.0f;
            for (int i = 0; i < k; ++i)
            {
                residual_weights[i] = target_token_ids[i] >= 0 ? target_probs[i] : 0.0f;
                residual_total += residual_weights[i];
            }
        }

        const float r = clamp_unit_threshold(residual_threshold) * residual_total;
        float cumulative = 0.0f;
        int selected = target_token_ids[0] >= 0 ? target_token_ids[0] : draft_token;
        for (int i = 0; i < k; ++i)
        {
            cumulative += residual_weights[i];
            if (r <= cumulative)
            {
                selected = target_token_ids[i] >= 0 ? target_token_ids[i] : selected;
                break;
            }
        }

        *out_token = selected;
        *out_accepted = 0;
    }

    LLAMINAR_SAMPLING_HD void speculative_verify_with_thresholds(
        const int *target_token_ids,
        const float *target_probs,
        const int *draft_token_ids,
        const float *draft_probs,
        int k,
        int draft_token,
        float accept_threshold,
        float residual_threshold,
        int *out_token,
        int *out_accepted,
        float *out_accept_probability,
        float *out_accept_threshold)
    {
        speculative_verify_with_thresholds_and_draft_probability(
            target_token_ids,
            target_probs,
            draft_token_ids,
            draft_probs,
            k,
            draft_token,
            0.0f,
            false,
            accept_threshold,
            residual_threshold,
            out_token,
            out_accepted,
            out_accept_probability,
            out_accept_threshold);
    }

    /**
     * @brief Verify a sampled greedy draft token against a compact target table.
     *
     * The vLLM-style greedy MTP proposal is a one-hot draft distribution:
     * `q(draft_token) = 1` and `q(other) = 0`.  The compact verifier can apply
     * the same rejection-sampling math without materializing a full draft
     * probability row.  On rejection, the residual distribution is therefore
     * the target distribution with the draft token removed.
     *
     * This helper deliberately uses the vLLM/no-draft acceptance convention
     * `threshold <= p/q`.  The older compact draft-table helper keeps its
     * historical strict comparison for backwards compatibility.
     */
    LLAMINAR_SAMPLING_HD void speculative_verify_with_thresholds_one_hot_draft(
        const int *target_token_ids,
        const float *target_probs,
        int k,
        int draft_token,
        float accept_threshold,
        float residual_threshold,
        int *out_token,
        int *out_accepted,
        float *out_accept_probability,
        float *out_accept_threshold)
    {
        const float p = distribution_probability(
            target_token_ids, target_probs, k, draft_token);
        const float accept_probability = speculative_accept_probability(p, 1.0f);
        const float threshold = clamp_unit_threshold(accept_threshold);

        if (out_accept_probability)
            *out_accept_probability = accept_probability;
        if (out_accept_threshold)
            *out_accept_threshold = threshold;

        if (threshold <= accept_probability)
        {
            *out_token = draft_token;
            *out_accepted = 1;
            return;
        }

        float residual_weights[kMaxTopK];
        float residual_total = 0.0f;
        for (int i = 0; i < k; ++i)
        {
            if (target_token_ids[i] < 0)
            {
                residual_weights[i] = 0.0f;
                continue;
            }

            const float q_i = target_token_ids[i] == draft_token ? 1.0f : 0.0f;
            const float w = fmaxf(0.0f, target_probs[i] - q_i);
            residual_weights[i] = w;
            residual_total += w;
        }

        if (!(residual_total > 0.0f))
        {
            residual_total = 0.0f;
            for (int i = 0; i < k; ++i)
            {
                residual_weights[i] = target_token_ids[i] >= 0 ? target_probs[i] : 0.0f;
                residual_total += residual_weights[i];
            }
        }

        const float r = clamp_unit_threshold(residual_threshold) * residual_total;
        float cumulative = 0.0f;
        int selected = target_token_ids[0] >= 0 ? target_token_ids[0] : draft_token;
        for (int i = 0; i < k; ++i)
        {
            cumulative += residual_weights[i];
            if (r <= cumulative)
            {
                selected = target_token_ids[i] >= 0 ? target_token_ids[i] : selected;
                break;
            }
        }

        *out_token = selected;
        *out_accepted = 0;
    }

    /**
     * @brief Verify one-hot greedy draft against a compact vLLM target table.
     *
     * This is the compact-table equivalent of the processed/full-probability
     * vLLM rejection kernels. Acceptance still uses `q(draft)=1`; rejection
     * samples the recovered token with inverse-exponential noise keyed by the
     * logical position and absolute vocabulary token id. Passing the full
     * vocabulary size preserves the same RNG offsets as the full-vocab path
     * even though this helper scans only the active compact support.
     */
    LLAMINAR_SAMPLING_HD void speculative_verify_with_thresholds_one_hot_draft_vllm_recovered(
        const int *target_token_ids,
        const float *target_probs,
        int k,
        int vocab_size,
        int draft_token,
        float accept_threshold,
        uint64_t inverse_sample_seed,
        int logical_position,
        int *out_token,
        int *out_accepted,
        float *out_accept_probability,
        float *out_accept_threshold)
    {
        const float p = distribution_probability(
            target_token_ids, target_probs, k, draft_token);
        const float accept_probability = speculative_accept_probability(p, 1.0f);
        const float threshold = clamp_unit_threshold(accept_threshold);

        if (out_accept_probability)
            *out_accept_probability = accept_probability;
        if (out_accept_threshold)
            *out_accept_threshold = threshold;

        if (threshold <= accept_probability)
        {
            *out_token = draft_token;
            *out_accepted = 1;
            return;
        }

        const uint64_t safe_position =
            static_cast<uint64_t>(logical_position > 0 ? logical_position : 0);
        const uint64_t safe_vocab =
            static_cast<uint64_t>(vocab_size > 0 ? vocab_size : k);
        float best_value = -1.0f;
        int best_token = -1;
        for (int i = 0; i < k; ++i)
        {
            const int token = target_token_ids[i];
            if (token < 0)
                continue;

            const float q_i = token == draft_token ? 1.0f : 0.0f;
            const float probability = fmaxf(0.0f, target_probs[i] - q_i);
            const uint64_t offset =
                safe_position * safe_vocab + static_cast<uint64_t>(token);
            const float uniform =
                uniform01(inverse_sample_seed ^ kInverseSampleDomain, offset);
            const float inverse_sample =
                inverse_exponential_from_uniform(uniform);
            const float value = probability * inverse_sample;
            if (value > best_value ||
                (value == best_value &&
                 (best_token < 0 || token < best_token)))
            {
                best_value = value;
                best_token = token;
            }
        }

        *out_token = best_token >= 0
                         ? best_token
                         : (target_token_ids[0] >= 0 ? target_token_ids[0] : draft_token);
        *out_accepted = 0;
    }

    /**
     * @brief Reduce row-wise speculative verifier decisions into one commit plan.
     *
     * Row kernels decide the stochastic accept/reject token independently. This
     * reducer applies the autoregressive semantics: emit tokens until the first
     * rejection or stop token, count the accepted speculative prefix, and expose
     * a ready token only when every verifier row accepted. The metadata layout
     * is fixed by SpeculativeBatchMetaIndex so host tests and GPU kernels cannot
     * drift.
     *
     * @param out_token_capacity Number of writable entries in `out_tokens`.
     *        The complete declared extent is initialized to `-1`, making the
     *        compact outcome byte-stable even when request rows share a larger
     *        configured batch stride.
     */
    LLAMINAR_SAMPLING_HD void summarize_speculative_verify_batch(
        int first_token,
        const int *row_tokens,
        const int *row_accepted,
        int row_count,
        const int *stop_tokens,
        int stop_token_count,
        int bonus_ready_token,
        int has_bonus_ready_token,
        int *out_tokens,
        int out_token_capacity,
        int *out_meta,
        const int *greedy_draft_tokens = nullptr)
    {
        if (!out_tokens || !out_meta ||
            row_count < 0 ||
            out_token_capacity < row_count + 1 ||
            stop_token_count < 0 ||
            stop_token_count > kSpeculativeBatchMaxStopTokens)
        {
            if (out_meta)
                out_meta[kSpecBatchMetaOk] = 0;
            return;
        }

        for (int i = 0; i < out_token_capacity; ++i)
            out_tokens[i] = -1;
        for (int i = 0; i < kSpeculativeBatchMetaCount; ++i)
            out_meta[i] = 0;

        if (first_token < 0)
        {
            out_meta[kSpecBatchMetaOk] = 0;
            return;
        }

        int output_count = 1;
        int consumed_rows = 0;
        int accepted_prefix = 0;
        int rejected_token = -1;
        bool stopped = false;
        for (int i = 0; i < stop_token_count; ++i)
        {
            if (stop_tokens && stop_tokens[i] == first_token)
            {
                stopped = true;
                break;
            }
        }
        bool all_accepted = true;
        out_tokens[0] = first_token;

        for (int row = 0; !stopped && row < row_count; ++row)
        {
            if (!row_tokens ||
                (!row_accepted && !greedy_draft_tokens) ||
                row_tokens[row] < 0)
            {
                out_meta[kSpecBatchMetaOk] = 0;
                return;
            }

            const int token = row_tokens[row];
            const bool accepted = row_accepted
                                      ? row_accepted[row] != 0
                                      : row_tokens[row] ==
                                            greedy_draft_tokens[row + 1];
            out_tokens[output_count++] = token;
            ++consumed_rows;

            if (accepted)
            {
                ++accepted_prefix;
            }
            else
            {
                all_accepted = false;
                rejected_token = token;
            }

            for (int i = 0; i < stop_token_count; ++i)
            {
                if (stop_tokens && stop_tokens[i] == token)
                {
                    stopped = true;
                    break;
                }
            }
            if (!accepted)
                break;
        }

        int ready_token = -1;
        int sampled_terminal = 0;
        if (!stopped && all_accepted)
        {
            if (!has_bonus_ready_token || bonus_ready_token < 0)
            {
                out_meta[kSpecBatchMetaOk] = 0;
                return;
            }
            ready_token = bonus_ready_token;
            sampled_terminal = 1;
        }

        const int commit_count =
            accepted_prefix + 1 < row_count + 1
                ? accepted_prefix + 1
                : row_count + 1;

        out_meta[kSpecBatchMetaOk] = 1;
        out_meta[kSpecBatchMetaOutputCount] = output_count;
        out_meta[kSpecBatchMetaAcceptedSpeculativePrefix] = accepted_prefix;
        out_meta[kSpecBatchMetaTargetVerifierStateCommitCount] = commit_count;
        out_meta[kSpecBatchMetaReadyToken] = ready_token;
        out_meta[kSpecBatchMetaRejectedVerifiedToken] = rejected_token;
        out_meta[kSpecBatchMetaStoppedOnOutput] = stopped ? 1 : 0;
        out_meta[kSpecBatchMetaAllSpeculativeAccepted] = all_accepted ? 1 : 0;
        out_meta[kSpecBatchMetaConsumedVerifierRows] = consumed_rows;
        out_meta[kSpecBatchMetaSampledTerminal] = sampled_terminal;
    }

    /**
     * @brief Reduce verifier rows without crossing a device-owned commit boundary.
     *
     * Dynamic MoE placement may change only between serial-visible decode
     * transactions. A grouped verifier can otherwise accept several rows and
     * carry execution past the exact token at which serial decode would run a
     * maintenance wave. This helper shortens the semantic transaction while
     * preserving the already-computed target samples and their logical
     * positions.
     *
     * If @p max_state_commit_rows is smaller than `row_count + 1`, at most
     * `max_state_commit_rows - 1` speculative rows are compared. The target
     * sample in the following verifier row becomes the ready condition token.
     * For example, a one-row commit budget emits only `first_token` and keeps
     * `row_tokens[0]` as the next condition. No row is resampled and no host
     * scalar participates in the decision.
     *
     * Rejection and stop-token semantics remain unchanged when either occurs
     * before the boundary. The ordinary terminal bonus is used only when the
     * complete declared verifier transaction fits inside the commit budget.
     *
     * @param max_state_commit_rows Positive number of verifier input states
     *        that may become visible before the next device maintenance edge.
     */
    LLAMINAR_SAMPLING_HD void
    summarize_speculative_verify_batch_at_commit_boundary(
        int first_token,
        const int *row_tokens,
        const int *row_accepted,
        int row_count,
        const int *stop_tokens,
        int stop_token_count,
        int bonus_ready_token,
        int has_bonus_ready_token,
        int max_state_commit_rows,
        int *out_tokens,
        int out_token_capacity,
        int *out_meta,
        const int *greedy_draft_tokens = nullptr,
        int leading_committed_output_count = 0)
    {
        if (max_state_commit_rows <= 0 ||
            leading_committed_output_count < 0 ||
            leading_committed_output_count > 1)
        {
            if (out_tokens && out_token_capacity > 0)
            {
                for (int i = 0; i < out_token_capacity; ++i)
                    out_tokens[i] = -1;
            }
            if (out_meta)
            {
                for (int i = 0; i < kSpeculativeBatchMetaCount; ++i)
                    out_meta[i] = 0;
            }
            return;
        }

        const int full_commit_rows = row_count + 1;
        const int physical_commit_budget =
            max_state_commit_rows >=
                    full_commit_rows - leading_committed_output_count
                ? full_commit_rows
                : max_state_commit_rows + leading_committed_output_count;
        const int effective_commit_rows =
            physical_commit_budget < full_commit_rows
                ? physical_commit_budget
                : full_commit_rows;
        const int effective_row_count = effective_commit_rows - 1;
        const bool stopped_at_commit_boundary =
            effective_row_count < row_count;
        const int effective_bonus_ready_token =
            stopped_at_commit_boundary && row_tokens
                ? row_tokens[effective_row_count]
                : bonus_ready_token;
        const int has_effective_bonus_ready_token =
            stopped_at_commit_boundary
                ? (row_tokens && effective_bonus_ready_token >= 0 ? 1 : 0)
                : has_bonus_ready_token;

        summarize_speculative_verify_batch(
            first_token,
            row_tokens,
            row_accepted,
            effective_row_count,
            stop_tokens,
            stop_token_count,
            effective_bonus_ready_token,
            has_effective_bonus_ready_token,
            out_tokens,
            out_token_capacity,
            out_meta,
            greedy_draft_tokens);

        if (out_meta && out_meta[kSpecBatchMetaOk] != 0)
        {
            out_meta[kSpecBatchMetaLeadingCommittedOutputCount] =
                leading_committed_output_count;
        }

        /*
         * A maintenance boundary is a third successful outcome category.  It
         * is neither a rejection nor acceptance of the complete physical
         * verifier batch.  The ordinary reducer above intentionally evaluates
         * only the serial-visible prefix so it can reuse the exact stop and
         * rejection semantics.  If that entire prefix accepted, reinterpret
         * its synthetic "bonus" as the already-sampled condition token at the
         * maintenance edge and make the distinction explicit in the ABI.
         */
        if (stopped_at_commit_boundary &&
            out_meta &&
            out_meta[kSpecBatchMetaOk] != 0 &&
            out_meta[kSpecBatchMetaStoppedOnOutput] == 0 &&
            out_meta[kSpecBatchMetaAllSpeculativeAccepted] != 0 &&
            out_meta[kSpecBatchMetaSampledTerminal] != 0)
        {
            out_meta[kSpecBatchMetaAllSpeculativeAccepted] = 0;
            out_meta[kSpecBatchMetaSampledTerminal] = 0;
            out_meta[kSpecBatchMetaCommitBoundaryClipped] = 1;
        }
    }

    /**
     * @brief Derive live-state publication rows from compact verifier metadata.
     *
     * The compact stochastic verifier summary intentionally has two different
     * counts:
     *
     * - kSpecBatchMetaAcceptedSpeculativePrefix counts accepted MTP draft rows.
     * - kSpecBatchMetaTargetVerifierStateCommitCount counts verifier input
     *   rows whose target-model state is mathematically valid. This includes row
     *   zero, the first main-model token.
     *
     * Accepted-state publication starts from the second count, then clamps it to
     * @p max_state_commit_rows. The clamp represents the serial-visible response
     * boundary: a terminal all-accepted verifier row can be valid speculative
     * evidence while its input token must remain the pending condition token.
     * Keeping this tiny helper shared between CPU tests and GPU kernels prevents
     * CUDA, ROCm, and CPU from drifting on rejection and response-budget
     * boundaries.
     */
    LLAMINAR_SAMPLING_HD void derive_speculative_publication_metadata(
        const int *meta,
        int meta_stride,
        int request_index,
        int padded_state_rows_per_request,
        int base_cached_tokens,
        int max_state_commit_rows,
        int *out_restore_row,
        int *out_target_cached_tokens,
        int *out_accepted_state_count,
        int *out_ok,
        const int32_t *output_tokens = nullptr,
        int output_token_stride = 0,
        int32_t *out_next_condition_token = nullptr,
        int *out_all_drafts_accepted = nullptr,
        int *out_stopped = nullptr)
    {
        if (out_restore_row)
            *out_restore_row = -1;
        if (out_target_cached_tokens)
            *out_target_cached_tokens = base_cached_tokens;
        if (out_accepted_state_count)
            *out_accepted_state_count = 0;
        if (out_next_condition_token)
            *out_next_condition_token = -1;
        if (out_all_drafts_accepted)
            *out_all_drafts_accepted = 0;
        if (out_stopped)
            *out_stopped = 0;
        if (out_ok)
            *out_ok = 0;

        if (!meta ||
            meta_stride < kSpeculativeBatchMetaCount ||
            request_index < 0 ||
            padded_state_rows_per_request <= 0 ||
            base_cached_tokens < 0 ||
            max_state_commit_rows < 0 ||
            max_state_commit_rows > padded_state_rows_per_request)
        {
            return;
        }

        const int *request_meta =
            meta + static_cast<size_t>(request_index) *
                       static_cast<size_t>(meta_stride);
        if (request_meta[kSpecBatchMetaOk] == 0)
            return;

        const int verifier_state_count =
            request_meta[kSpecBatchMetaTargetVerifierStateCommitCount];
        if (verifier_state_count < 0 ||
            verifier_state_count > padded_state_rows_per_request)
        {
            return;
        }
        const int accepted_state_count =
            verifier_state_count < max_state_commit_rows
                ? verifier_state_count
                : max_state_commit_rows;

        if (out_accepted_state_count)
            *out_accepted_state_count = accepted_state_count;
        if (out_target_cached_tokens)
            *out_target_cached_tokens =
                base_cached_tokens + accepted_state_count;
        if (out_all_drafts_accepted)
            *out_all_drafts_accepted =
                request_meta[kSpecBatchMetaAllSpeculativeAccepted] != 0 ? 1 : 0;
        if (out_stopped)
            *out_stopped =
                request_meta[kSpecBatchMetaStoppedOnOutput] != 0 ? 1 : 0;
        if (out_next_condition_token && output_tokens && output_token_stride > 0)
        {
            const int ready_token =
                request_meta[kSpecBatchMetaReadyToken];
            const bool sampled_terminal =
                request_meta[kSpecBatchMetaSampledTerminal] != 0;
            const bool commit_boundary_clipped =
                request_meta[kSpecBatchMetaCommitBoundaryClipped] != 0;
            const int output_count =
                request_meta[kSpecBatchMetaOutputCount];
            const bool publication_clipped =
                accepted_state_count < verifier_state_count;
            if (publication_clipped &&
                accepted_state_count >= 0 &&
                accepted_state_count < output_count &&
                accepted_state_count < output_token_stride)
            {
                /*
                 * The first verifier output beyond the published state prefix is
                 * the exact serial condition token.  A sampled terminal token is
                 * one generation farther ahead and must not cross this boundary.
                 */
                *out_next_condition_token =
                    output_tokens[static_cast<size_t>(request_index) *
                                      static_cast<size_t>(output_token_stride) +
                                  static_cast<size_t>(accepted_state_count)];
            }
            else if ((sampled_terminal || commit_boundary_clipped) &&
                     ready_token >= 0)
            {
                *out_next_condition_token = ready_token;
            }

            if (*out_next_condition_token < 0 &&
                output_count > 0 && output_count <= output_token_stride)
            {
                *out_next_condition_token =
                    output_tokens[static_cast<size_t>(request_index) *
                                      static_cast<size_t>(output_token_stride) +
                                  static_cast<size_t>(output_count - 1)];
            }
        }
        if (out_restore_row && accepted_state_count > 0)
        {
            *out_restore_row =
                request_index * padded_state_rows_per_request +
                accepted_state_count - 1;
        }
        if (out_ok)
            *out_ok = 1;
    }

    /**
     * @brief Commit one compact outcome and derive its live-state publication.
     *
     * GPU generation used to launch one scalar kernel to append response tokens
     * and a second scalar kernel to derive the accepted KV/recurrent-state row.
     * Both operations consume the same compact token/metadata row, run on the
     * same producer stream, and form one indivisible serial-decode transaction.
     * Keeping them in separate launches added latency and allowed later callers
     * to accidentally publish state without first committing the response
     * ledger.  This shared helper makes that illegal by construction.
     *
     * Publication is validated before response bytes are mutated.  Response
     * append is then all-or-nothing: its helper validates every count and
     * capacity before copying tokens.  If either half fails, @p out_ok is zero
     * and the resident controller becomes terminal, so downstream publication
     * kernels cannot expose a partially committed transaction.
     *
     * @param output_tokens Compact output rows for every request.
     * @param output_token_stride Compact token capacity per request.
     * @param meta Compact metadata rows for every request.
     * @param meta_stride Compact metadata capacity per request.
     * @param request_index Request row owned by this invocation.
     * @param padded_state_rows_per_request Static verifier graph row capacity.
     * @param base_cached_tokens Device-owned cache count before verifier replay.
     * @param response_tokens Persistent response row for this request.
     * @param response_capacity Persistent response-row capacity.
     * @param control Persistent generation-controller row.  Its current
     *        transaction budget is the sole publication limit.
     * @return true only when response and publication state were both committed.
     */
    LLAMINAR_SAMPLING_HD bool
    commit_device_generation_and_derive_speculative_publication_metadata(
        const int32_t *output_tokens,
        int output_token_stride,
        int *meta,
        int meta_stride,
        int request_index,
        int padded_state_rows_per_request,
        int base_cached_tokens,
        int32_t *response_tokens,
        int response_capacity,
        int *control,
        int *out_restore_row,
        int *out_target_cached_tokens,
        int *out_accepted_state_count,
        int *out_ok,
        int32_t *out_next_condition_token = nullptr,
        int *out_all_drafts_accepted = nullptr,
        int *out_stopped = nullptr)
    {
        /*
         * Validate every required address and geometry before either helper can
         * derive a request-row address.  Backend launchers enforce the same
         * contract on the host, but this device-side guard is still essential:
         * graph node parameters are mutable, and a malformed replay must become
         * a terminal request error instead of performing undefined pointer
         * arithmetic or publishing stale metadata.
         */
        if (!output_tokens ||
            output_token_stride <= 0 ||
            !meta ||
            meta_stride < kSpeculativeBatchMetaCount ||
            request_index < 0 ||
            padded_state_rows_per_request <= 0 ||
            base_cached_tokens < 0 ||
            !response_tokens ||
            response_capacity <= 0 ||
            !control ||
            !out_restore_row ||
            !out_target_cached_tokens ||
            !out_accepted_state_count ||
            !out_ok)
        {
            return fail_device_generation_control(
                control,
                DeviceGenerationError::InvalidPublicationMetadata);
        }

        /*
         * A successfully completed request is an absorbing state.  A static
         * graph may reach this node again before its device-side loop observes
         * the terminal predicate, but stale verifier scratch must never turn a
         * successful terminal controller into an error or republish an old
         * accepted row.  Publish an explicitly inert state transaction while
         * preserving the last valid condition-token value: speculative readers
         * can safely drain already-enqueued work, while every live-state
         * publisher sees `out_ok == 0` and a negative restore row.
         *
         * This check deliberately precedes every compact-metadata read.  Once
         * terminal, those bytes are outside the request state machine and are
         * neither trusted nor repaired.
         */
        if (control[kDeviceGenerationControlOk] != 0 &&
            control[kDeviceGenerationControlRequestComplete] != 0)
        {
            *out_restore_row = -1;
            *out_target_cached_tokens = base_cached_tokens;
            *out_accepted_state_count = 0;
            *out_ok = 0;
            if (out_all_drafts_accepted)
                *out_all_drafts_accepted = 0;
            if (out_stopped)
                *out_stopped = 1;
            return true;
        }

        *out_ok = 0;
        if (control[kDeviceGenerationControlOk] == 0)
            return false;

        int *request_meta =
            meta + static_cast<size_t>(request_index) *
                       static_cast<size_t>(meta_stride);
        const int transaction_commit_budget =
            control[kDeviceGenerationControlTransactionCommitBudget];

        derive_speculative_publication_metadata(
            meta,
            meta_stride,
            request_index,
            padded_state_rows_per_request,
            base_cached_tokens,
            transaction_commit_budget,
            out_restore_row,
            out_target_cached_tokens,
            out_accepted_state_count,
            out_ok,
            output_tokens,
            output_token_stride,
            out_next_condition_token,
            out_all_drafts_accepted,
            out_stopped);

        if (!out_ok || *out_ok == 0)
        {
            request_meta[kSpecBatchMetaOk] = 0;
            return fail_device_generation_control(
                control,
                DeviceGenerationError::InvalidPublicationMetadata);
        }

        const int32_t *request_output_tokens =
            output_tokens + static_cast<size_t>(request_index) *
                                static_cast<size_t>(output_token_stride);
        if (!append_speculative_outcome_to_device_generation(
                request_output_tokens,
                output_token_stride,
                request_meta,
                meta_stride,
                response_tokens,
                response_capacity,
                control))
        {
            *out_ok = 0;
            request_meta[kSpecBatchMetaOk] = 0;
            return false;
        }
        return true;
    }

    /**
     * @brief Derive shifted MTP-KV counts from canonical primary publication.
     *
     * The primary publication transaction has already validated compact
     * verifier metadata and applied the resident response/maintenance budget.
     * Re-reading compact metadata here would repeat that policy with another
     * scalar input and could let main and shifted caches publish different
     * prefixes.  This helper therefore accepts only the canonical primary
     * output: base count, committed target count, and publication validity.
     *
     * Depth `d` is `d + 1` rows behind the main model.  The shifted target is
     * `max(0, main_target - d - 1)`, while its accepted count is the delta from
     * the correspondingly shifted pre-transaction base.  This preserves ring
     * head movement at short prefixes without inventing a second commit limit.
     */
    LLAMINAR_SAMPLING_HD void
    derive_shifted_speculative_publication_metadata_from_primary(
        int base_cached_tokens,
        int main_target_cached_tokens,
        int main_publication_ok,
        int mtp_depth,
        int *out_target_cached_tokens,
        int *out_accepted_state_count,
        int *out_ok)
    {
        if (out_target_cached_tokens)
            *out_target_cached_tokens = 0;
        if (out_accepted_state_count)
            *out_accepted_state_count = 0;
        if (out_ok)
            *out_ok = 0;

        if (mtp_depth < 0 ||
            base_cached_tokens < 0 ||
            main_target_cached_tokens < base_cached_tokens ||
            main_publication_ok == 0)
            return;

        const int shift = mtp_depth + 1;
        const int base_shifted =
            base_cached_tokens > shift ? base_cached_tokens - shift : 0;
        const int target_shifted =
            main_target_cached_tokens > shift
                ? main_target_cached_tokens - shift
                : 0;
        const int accepted_shifted =
            target_shifted >= base_shifted
                ? target_shifted - base_shifted
                : 0;

        if (out_target_cached_tokens)
            *out_target_cached_tokens = target_shifted;
        if (out_accepted_state_count)
            *out_accepted_state_count = accepted_shifted;
        if (out_ok)
            *out_ok = 1;
    }

    /**
     * @brief Prepare valid shifted-MTP suffix tokens without reading metadata on host.
     *
     * LocalTP device-resident publication may need to append shifted sidecar KV
     * rows for accepted verifier outputs after the first sidecar-owned row.  The
     * accepted-state count is stored in compact device metadata, so GPU callers
     * use this helper to build a fixed-shape sidecar token rowset while preserving
     * the exact serial boundary:
     *
     * - rows before `accepted_state_count - first_output_token_index` copy the
     *   matching compact output token,
     * - rows beyond that boundary receive @p filler_token and are discarded by the
     *   later shifted-KV publication count,
     * - invalid metadata never copies speculative output tokens.
     */
    LLAMINAR_SAMPLING_HD void prepare_speculative_shifted_kv_tokens(
        const int *meta,
        int meta_stride,
        const int32_t *output_tokens,
        int output_token_stride,
        int request_index,
        int first_output_token_index,
        int row_count,
        int32_t filler_token,
        int32_t *out_tokens)
    {
        if (!out_tokens || row_count <= 0)
            return;

        for (int row = 0; row < row_count; ++row)
            out_tokens[row] = filler_token;

        if (!meta ||
            !output_tokens ||
            meta_stride < kSpeculativeBatchMetaCount ||
            output_token_stride <= first_output_token_index ||
            request_index < 0 ||
            first_output_token_index < 0)
        {
            return;
        }

        const int *request_meta =
            meta + static_cast<size_t>(request_index) *
                       static_cast<size_t>(meta_stride);
        if (request_meta[kSpecBatchMetaOk] == 0)
            return;

        const int accepted_state_count =
            request_meta[kSpecBatchMetaTargetVerifierStateCommitCount];
        const int output_count = request_meta[kSpecBatchMetaOutputCount];
        const int32_t *request_tokens =
            output_tokens + static_cast<size_t>(request_index) *
                                static_cast<size_t>(output_token_stride);
        if (output_count < 0 || output_count > output_token_stride)
            return;
        if (output_count > 0)
        {
            const int32_t live_filler = request_tokens[0];
            for (int row = 0; row < row_count; ++row)
                out_tokens[row] = live_filler;
        }
        if (accepted_state_count <= first_output_token_index ||
            output_count <= first_output_token_index)
        {
            return;
        }

        const int accepted_suffix_rows =
            accepted_state_count - first_output_token_index;
        const int output_suffix_rows =
            output_count - first_output_token_index;
        const int copy_rows =
            accepted_suffix_rows < output_suffix_rows
                ? accepted_suffix_rows
                : output_suffix_rows;
        const int bounded_copy_rows =
            copy_rows < row_count ? copy_rows : row_count;
        for (int row = 0; row < bounded_copy_rows; ++row)
        {
            out_tokens[row] =
                request_tokens[first_output_token_index + row];
        }
    }

    /**
     * @brief Decide whether a speculative batch needs a bonus ready token.
     *
     * GPU lazy verifier kernels use this before sampling the bonus distribution:
     * if the first token stops, any verifier row stops, or any verifier row
     * rejects, the bonus token is not semantically consumed and should not burn
     * a stochastic sample.  The full reducer still owns the final metadata; this
     * helper only answers the cheap "is bonus needed?" question from the same
     * shared semantics.
     */
    LLAMINAR_SAMPLING_HD bool speculative_batch_needs_bonus_ready_token(
        int first_token,
        const int *row_tokens,
        const int *row_accepted,
        int row_count,
        const int *stop_tokens,
        int stop_token_count)
    {
        if (first_token < 0 ||
            row_count < 0 ||
            stop_token_count < 0 ||
            stop_token_count > kSpeculativeBatchMaxStopTokens)
        {
            return false;
        }

        for (int i = 0; i < stop_token_count; ++i)
        {
            if (stop_tokens && stop_tokens[i] == first_token)
                return false;
        }

        for (int row = 0; row < row_count; ++row)
        {
            if (!row_tokens || !row_accepted || row_tokens[row] < 0)
                return false;

            const int token = row_tokens[row];
            if (row_accepted[row] == 0)
                return false;

            for (int i = 0; i < stop_token_count; ++i)
            {
                if (stop_tokens && stop_tokens[i] == token)
                    return false;
            }
        }

        return true;
    }

    /**
     * @brief Summarize greedy speculative verifier rows from device tokens.
     *
     * `verifier_tokens[row]` is the target-model greedy token for verifier row
     * `row`. `draft_tokens[0]` is the first already-sampled target token and
     * `draft_tokens[row + 1]` is the speculative draft token checked by
     * `verifier_tokens[row]`. `verifier_tokens[compare_row_count]` is the bonus
     * ready token consumed only when every speculative row accepts.
     */
    LLAMINAR_SAMPLING_HD void summarize_greedy_speculative_verify_batch(
        int first_token,
        const int *verifier_tokens,
        const int *draft_tokens,
        int compare_row_count,
        const int *stop_tokens,
        int stop_token_count,
        int *out_tokens,
        int out_token_capacity,
        int *out_meta)
    {
        if (!verifier_tokens || !draft_tokens || compare_row_count < 0)
        {
            if (out_meta)
                out_meta[kSpecBatchMetaOk] = 0;
            return;
        }

        const int bonus_ready_token = verifier_tokens[compare_row_count];
        summarize_speculative_verify_batch(
            first_token,
            verifier_tokens,
            /*row_accepted=*/nullptr,
            compare_row_count,
            stop_tokens,
            stop_token_count,
            bonus_ready_token,
            /*has_bonus_ready_token=*/1,
            out_tokens,
            out_token_capacity,
            out_meta,
            draft_tokens);
    }

    /**
     * @brief Greedy companion to the device-owned commit-boundary reducer.
     *
     * `verifier_tokens[compare_row_count]` remains the ordinary bonus sample.
     * When the maintenance boundary is earlier, the shared boundary reducer
     * instead promotes `verifier_tokens[max_state_commit_rows - 1]` to the
     * ready condition token without changing its value or logical position.
     */
    LLAMINAR_SAMPLING_HD void
    summarize_greedy_speculative_verify_batch_at_commit_boundary(
        int first_token,
        const int *verifier_tokens,
        const int *draft_tokens,
        int compare_row_count,
        const int *stop_tokens,
        int stop_token_count,
        int max_state_commit_rows,
        int *out_tokens,
        int out_token_capacity,
        int *out_meta,
        int leading_committed_output_count = 0)
    {
        if (!verifier_tokens || !draft_tokens || compare_row_count < 0)
        {
            if (out_meta)
                out_meta[kSpecBatchMetaOk] = 0;
            return;
        }

        summarize_speculative_verify_batch_at_commit_boundary(
            first_token,
            verifier_tokens,
            /*row_accepted=*/nullptr,
            compare_row_count,
            stop_tokens,
            stop_token_count,
            verifier_tokens[compare_row_count],
            /*has_bonus_ready_token=*/1,
            max_state_commit_rows,
            out_tokens,
            out_token_capacity,
            out_meta,
            draft_tokens,
            leading_committed_output_count);
    }

} // namespace llaminar2::sampling_math

#undef LLAMINAR_SAMPLING_HD
