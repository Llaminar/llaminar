/**
 * @file MTPParitySnapshotContext.h
 * @brief Typed ownership and parsing rules for production MTP checkpoint keys.
 *
 * CPU sidecars consume host-owned condition tokens and positions, whereas GPU
 * sidecars consume one of several device-owned live-state publications.  The
 * snapshot qualifier records that authority.  Parity code uses this helper to
 * select the context that actually ran and to distinguish model stages from
 * operational state snapshots that share the `MTP_DECODE_SIDECAR_` prefix.
 */

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace llaminar2::test::parity
{
    /**
     * @brief Input authority and replay role of one production MTP sidecar.
     */
    enum class MTPParityCheckpointContext
    {
        HostConditionTokenLivePosition,
        HostChainedDraftLivePosition,
        DeviceTargetTokenLivePosition,
        DeviceResidentLogicalState,
        DeviceChainedTokenLivePosition,
    };

    /**
     * @brief Semantic position of one retained sidecar snapshot in a transaction.
     *
     * The production predictor owns two snapshot banks: the primary sidecar and
     * one reusable chained-sidecar bank.  A complete retained transaction may
     * execute the chained graph many times, so that second bank represents the
     * terminal recursive row, not a response-budget-derived intermediate row.
     */
    enum class MTPParityCheckpointRole
    {
        Primary,
        TerminalChained,
    };

    /** @brief Exact runtime context and Hugging Face row for one checkpoint bank. */
    struct MTPParityCheckpointIdentity
    {
        MTPParityCheckpointRole role = MTPParityCheckpointRole::Primary;
        MTPParityCheckpointContext context =
            MTPParityCheckpointContext::HostConditionTokenLivePosition;
        int reference_depth = -1;
    };

    /**
     * @brief Exact namespaces of one numerically compared production MTP bank.
     *
     * A sidecar comparison is reported at a synthetic model-layer index, but
     * its live snapshot does not use a guessed `layerN_` key. The retained
     * graph publishes through a typed runtime-context prefix while the
     * Hugging Face pack uses a depth/branch-qualified prefix. Carrying both
     * authorities through the comparison callback prevents evidence collectors
     * from reconstructing either namespace from the synthetic layer number.
     */
    struct ComparedMTPParityCheckpoint
    {
        int reference_step = -1; ///< Main decode row owning the transaction.
        int model_layer = -1;    ///< Synthetic sidecar layer in parity CSVs.
        MTPParityCheckpointIdentity identity; ///< Runtime role/reference depth.
        std::string production_stage_prefix;  ///< Live prefix ending `MTP0_`.
        std::string reference_stage_prefix;   ///< HF prefix ending `MTPD_`.

        /** @return Whether this identity names both checkpoint authorities. */
        [[nodiscard]] bool valid() const noexcept
        {
            return reference_step >= 0 && model_layer >= 0 &&
                   identity.reference_depth >= 0 &&
                   !production_stage_prefix.empty() &&
                   production_stage_prefix.back() == '_' &&
                   !reference_stage_prefix.empty() &&
                   reference_stage_prefix.back() == '_';
        }

        /** @return Exact production key for one semantic stage suffix. */
        [[nodiscard]] std::string productionKey(
            std::string_view semantic_stage) const
        {
            return production_stage_prefix + std::string(semantic_stage);
        }

        /** @return Exact Hugging Face key for one semantic stage suffix. */
        [[nodiscard]] std::string referenceKey(
            std::string_view semantic_stage) const
        {
            return reference_stage_prefix + std::string(semantic_stage);
        }
    };

    /**
     * @brief Bounded checkpoint plan for one production MTP transaction.
     *
     * Depth one has only the primary row. Deeper transactions retain both the
     * primary row and the final chained row. Intermediate recursive values flow
     * through the same chained executable into the terminal checkpoint and the
     * exact serial-token trajectory; independently generated fixed-depth cells
     * certify every declared matrix depth. A row is never inferred from the
     * caller's response limit.
     */
    struct MTPParityCheckpointPlan
    {
        std::array<MTPParityCheckpointIdentity, 2> checkpoints{};
        size_t count = 0;

        /** @return True when every published identity has a legal reference row. */
        [[nodiscard]] constexpr bool valid() const noexcept
        {
            if (count == 0 || count > checkpoints.size())
                return false;
            for (size_t index = 0; index < count; ++index)
            {
                if (checkpoints[index].reference_depth < 0)
                    return false;
            }
            return true;
        }
    };

    /**
     * @brief Build the exact additive Hugging Face branch checkpoint prefix.
     *
     * Canonical recursive snapshots are named `decode_stepS_MTPD_*`. When a
     * quantized production predictor chooses a different token, its tensors
     * are comparable only with a forced-branch oracle whose filename embeds
     * every condition token consumed before depth D. Keeping this spelling in
     * one typed helper prevents generators and comparisons from silently
     * disagreeing about whether the current depth's output token belongs in
     * the branch identity (it does not).
     *
     * @param reference_step Authenticated main-model decode step.
     * @param reference_depth Recursive MTP row being compared.
     * @param condition_tokens Exact preceding device/host-owned draft tokens;
     *        its size must equal @p reference_depth.
     * @return Prefix ending in `_MTPD_`, ready for a stage suffix.
     * @throws std::invalid_argument for an invalid step/depth/token identity.
     */
    [[nodiscard]] inline std::string mtpParityBranchReferencePrefix(
        int reference_step,
        int reference_depth,
        std::span<const int32_t> condition_tokens)
    {
        if (reference_step < 0 || reference_depth <= 0 ||
            condition_tokens.size() !=
                static_cast<size_t>(reference_depth))
        {
            throw std::invalid_argument(
                "MTP branch reference identity has invalid step/depth geometry");
        }

        std::string prefix =
            "decode_step" + std::to_string(reference_step) + "_BRANCH";
        for (const int32_t token : condition_tokens)
        {
            if (token < 0)
            {
                throw std::invalid_argument(
                    "MTP branch reference identity contains a negative token");
            }
            prefix += "_" + std::to_string(token);
        }
        prefix += "_MTP" + std::to_string(reference_depth) + "_";
        return prefix;
    }

    /**
     * @brief Resolve checkpoint identity from executed graph geometry.
     *
     * @param cpu_owned Whether the production sidecar consumes host-owned state.
     * @param execution_draft_depth Actual depth selected for the externally
     *        materialized transaction whose snapshot banks are published.
     * @return Primary/terminal checkpoint plan, or an invalid empty plan.
     *
     * Response capacity is deliberately absent. It limits visible commit rows,
     * but it does not rename a retained graph invocation or its snapshot bytes.
     */
    [[nodiscard]] constexpr MTPParityCheckpointPlan
    makeMTPParityCheckpointPlan(
        bool cpu_owned,
        int execution_draft_depth) noexcept
    {
        MTPParityCheckpointPlan plan;
        if (execution_draft_depth <= 0)
            return plan;

        plan.checkpoints[0] = {
            .role = MTPParityCheckpointRole::Primary,
            .context = cpu_owned
                           ? MTPParityCheckpointContext::
                                 HostConditionTokenLivePosition
                           : MTPParityCheckpointContext::
                                 DeviceTargetTokenLivePosition,
            .reference_depth = 0,
        };
        plan.count = 1;
        if (execution_draft_depth == 1)
            return plan;

        plan.checkpoints[1] = {
            .role = MTPParityCheckpointRole::TerminalChained,
            .context = cpu_owned
                           ? MTPParityCheckpointContext::
                                 HostChainedDraftLivePosition
                           : MTPParityCheckpointContext::
                                 DeviceChainedTokenLivePosition,
            .reference_depth = execution_draft_depth - 1,
        };
        plan.count = 2;
        return plan;
    }

    /**
     * @brief Result of comparing production MTP counters around decodeStep().
     */
    enum class MTPParityTransactionActivity
    {
        None,
        Speculative,
        Inconsistent,
    };

    /** @brief Minimal monotonic evidence published by a production MTP loop. */
    struct MTPParityTransactionCounters
    {
        uint64_t draft_steps = 0;
        uint64_t verifier_runs = 0;
    };

    /**
     * @brief Immutable placement epoch inherited by one complete request.
     *
     * Recurrent GDN, short-convolution, and KV state encodes arithmetic from
     * every preceding token. A final row's execution epoch is therefore not
     * sufficient serial-equivalence evidence after expert movement. This value
     * remains engaged only while prefill and every later production boundary
     * begin, execute, and retire in one identical residency epoch.
     */
    struct MTPParityPlacementEpochTrajectory
    {
        std::optional<uint64_t> epoch;

        /**
         * @brief Extend the request history with one authenticated boundary.
         *
         * Once invalidated, a trajectory remains invalid: a stable suffix does
         * not erase state inherited from a different earlier placement.
         */
        constexpr void observe(
            uint64_t movement_epoch_begin,
            uint64_t movement_epoch_end,
            uint64_t execution_epoch) noexcept
        {
            if (!epoch.has_value() || execution_epoch == 0u ||
                movement_epoch_begin != movement_epoch_end ||
                movement_epoch_begin != execution_epoch ||
                *epoch != execution_epoch)
            {
                epoch.reset();
            }
        }

        /** @brief Mark inherited placement provenance as unavailable. */
        constexpr void invalidate() noexcept
        {
            epoch.reset();
        }

        /** @return Whether the complete request history used @p candidate. */
        [[nodiscard]] constexpr bool matches(uint64_t candidate) const noexcept
        {
            return candidate != 0u && epoch == candidate;
        }
    };

    /**
     * @brief Exact grouped-response comparison against a serial Llaminar oracle.
     *
     * Hugging Face remains the numerical checkpoint oracle, but a quantized
     * near-tie may legitimately select a different local top-1 token. Grouped
     * MTP therefore proves token identity against serial execution of the same
     * Llaminar model, topology, and sampling policy. This result keeps that
     * distinction typed and reports the first broken response edge.
     */
    struct MTPParityTokenComparison
    {
        bool exact = false;
        size_t compared_tokens = 0;
        size_t mismatch_index = std::numeric_limits<size_t>::max();
        int32_t serial_token = -1;
        int32_t grouped_token = -1;
    };

    /**
     * @brief Reference row whose first speculative token is serially correct.
     *
     * The row is selected from immutable Hugging Face evidence before the
     * production request starts.  It lets one response-bounded transaction prove
     * both its retained checkpoint bank and a visible accepted draft without
     * launching a second inference lifecycle.
     */
    struct MTPParityAcceptedDraftReference
    {
        size_t reference_step = 0; ///< Decode-step identity of the main row.
        int32_t base_token = -1; ///< Serial token emitted before the draft.
        int32_t first_draft_token = -1; ///< Matching serial/HF draft token.

        /** @return Whether both token identities are initialized. */
        [[nodiscard]] constexpr bool valid() const noexcept
        {
            return base_token >= 0 && first_draft_token >= 0;
        }
    };

    /**
     * @brief One ordinary decode row used to prove a serial token edge.
     *
     * Teacher forcing is a valid free-running serial oracle only while every
     * forced token equals the predecessor row's production argmax.  Carrying
     * both sides of that edge prevents a numerically compared but divergent
     * teacher-forced trajectory from being reused by the grouped-MTP proof.
     */
    struct MTPParityCertifiedDecodeRow
    {
        size_t reference_step = 0; ///< Contiguous row identity, beginning at zero.
        int32_t committed_token = -1; ///< Token actually forwarded by this row.
        int32_t predicted_successor_token = -1; ///< Production argmax after it.
    };

    /** @brief Why ordinary decode could not certify a requested serial horizon. */
    enum class MTPParitySerialCertificationFailure
    {
        None,
        InvalidRequestedHorizon,
        InvalidPrefillPrediction,
        MissingDecodeRow,
        NonContiguousDecodeRow,
        DiscontinuousTokenEdge,
        InvalidSuccessorPrediction,
    };

    /** @return Stable diagnostic spelling for one certification outcome. */
    [[nodiscard]] constexpr std::string_view
    mtpParitySerialCertificationFailureName(
        MTPParitySerialCertificationFailure failure) noexcept
    {
        switch (failure)
        {
        case MTPParitySerialCertificationFailure::None:
            return "none";
        case MTPParitySerialCertificationFailure::InvalidRequestedHorizon:
            return "invalid_requested_horizon";
        case MTPParitySerialCertificationFailure::InvalidPrefillPrediction:
            return "invalid_prefill_prediction";
        case MTPParitySerialCertificationFailure::MissingDecodeRow:
            return "missing_decode_row";
        case MTPParitySerialCertificationFailure::NonContiguousDecodeRow:
            return "non_contiguous_decode_row";
        case MTPParitySerialCertificationFailure::DiscontinuousTokenEdge:
            return "discontinuous_token_edge";
        case MTPParitySerialCertificationFailure::InvalidSuccessorPrediction:
            return "invalid_successor_prediction";
        }
        return "unknown";
    }

    /**
     * @brief Typed result of deriving free-running serial tokens from decode.
     */
    struct MTPParitySerialTrajectoryCertification
    {
        std::vector<int32_t> tokens; ///< Exact certified output prefix.
        MTPParitySerialCertificationFailure failure =
            MTPParitySerialCertificationFailure::None;
        size_t failure_row = std::numeric_limits<size_t>::max();
        int32_t expected_token = -1;
        int32_t observed_token = -1;

        /** @return Whether the complete requested output horizon was proven. */
        [[nodiscard]] bool complete(size_t required_output_count) const noexcept
        {
            return required_output_count > 0u &&
                   failure == MTPParitySerialCertificationFailure::None &&
                   tokens.size() == required_output_count;
        }
    };

    /**
     * @brief Certify that teacher-forced rows equal free-running serial decode.
     *
     * The prefill argmax is serial output zero. Decode row zero must consume
     * that exact token, then its argmax becomes serial output one, and so on.
     * This induction is byte-token exact; Hugging Face remains the independent
     * tensor oracle but is not substituted for Llaminar's serial trajectory.
     * A missing row is distinguishable from a broken edge so callers may run
     * an explicitly longer oracle only when the compared corpus is too short.
     *
     * @param prefill_predicted_token Production argmax after prompt prefill.
     * @param rows Ordered authenticated production decode rows.
     * @param required_output_count Number of serial output tokens required.
     * @return Certified prefix and an explicit terminal status.
     */
    [[nodiscard]] inline MTPParitySerialTrajectoryCertification
    certifyMTPParitySerialTrajectory(
        int32_t prefill_predicted_token,
        std::span<const MTPParityCertifiedDecodeRow> rows,
        size_t required_output_count)
    {
        MTPParitySerialTrajectoryCertification result;
        if (required_output_count == 0u)
        {
            result.failure =
                MTPParitySerialCertificationFailure::InvalidRequestedHorizon;
            return result;
        }
        if (prefill_predicted_token < 0)
        {
            result.failure =
                MTPParitySerialCertificationFailure::InvalidPrefillPrediction;
            return result;
        }

        result.tokens.reserve(required_output_count);
        result.tokens.push_back(prefill_predicted_token);
        while (result.tokens.size() < required_output_count)
        {
            const size_t row_index = result.tokens.size() - 1u;
            if (row_index >= rows.size())
            {
                result.failure =
                    MTPParitySerialCertificationFailure::MissingDecodeRow;
                result.failure_row = row_index;
                return result;
            }

            const MTPParityCertifiedDecodeRow &row = rows[row_index];
            if (row.reference_step != row_index)
            {
                result.failure =
                    MTPParitySerialCertificationFailure::NonContiguousDecodeRow;
                result.failure_row = row_index;
                result.expected_token = static_cast<int32_t>(row_index);
                result.observed_token = static_cast<int32_t>(row.reference_step);
                return result;
            }
            if (row.committed_token != result.tokens.back())
            {
                result.failure =
                    MTPParitySerialCertificationFailure::DiscontinuousTokenEdge;
                result.failure_row = row_index;
                result.expected_token = result.tokens.back();
                result.observed_token = row.committed_token;
                return result;
            }
            if (row.predicted_successor_token < 0)
            {
                result.failure = MTPParitySerialCertificationFailure::
                    InvalidSuccessorPrediction;
                result.failure_row = row_index;
                return result;
            }
            result.tokens.push_back(row.predicted_successor_token);
        }
        return result;
    }

    /**
     * @brief Execution and response geometry for one parity checkpoint transaction.
     *
     * GPU generation owns its response ledger on device. A two-token public
     * budget admits one condition row plus one speculative output, while the
     * device controller deliberately retains the complete selected depth-N
     * graph geometry and clips only publication at the commit boundary. That
     * is the smallest production request which executes one grouped verifier
     * transaction and cannot advance into a second transaction.
     *
     * CPU generation has no resident controller to own that separation, so its
     * public budget remains the complete condition-plus-drafts width.
     */
    struct MTPParityCheckpointTransactionPlan
    {
        int execution_draft_depth = 0; ///< Captured/selected predictor width.
        int response_token_budget = 0; ///< Public decodeStep response limit.
        bool device_commit_boundary = false; ///< Device clips publication only.

        /** @return Whether the plan describes one legal speculative transaction. */
        [[nodiscard]] constexpr bool valid() const noexcept
        {
            return execution_draft_depth > 0 && response_token_budget > 1 &&
                   (!device_commit_boundary || response_token_budget == 2);
        }
    };

    /**
     * @brief Plan one checkpoint transaction without changing graph geometry.
     *
     * @param device_controller_owned Whether a GPU resident controller owns
     *        response/state publication.
     * @param execution_draft_depth Positive selected MTP graph depth.
     * @return A valid response/execution plan, or an invalid zero plan.
     */
    [[nodiscard]] constexpr MTPParityCheckpointTransactionPlan
    makeMTPParityCheckpointTransactionPlan(
        bool device_controller_owned,
        int execution_draft_depth) noexcept
    {
        if (execution_draft_depth <= 0 ||
            execution_draft_depth == std::numeric_limits<int>::max())
        {
            return {};
        }
        return {
            .execution_draft_depth = execution_draft_depth,
            .response_token_budget =
                device_controller_owned ? 2 : execution_draft_depth + 1,
            .device_commit_boundary = device_controller_owned,
        };
    }

    /**
     * @brief Find the earliest reference row with one provably accepted draft.
     *
     * @param serial_tokens Ordered reference decode tokens.
     * @param first_draft_tokens MTP0 argmax at each aligned decode step.
     * @return The first row where MTP0 predicts the following serial token.
     */
    [[nodiscard]] constexpr std::optional<MTPParityAcceptedDraftReference>
    firstMTPParityAcceptedDraftReference(
        std::span<const int32_t> serial_tokens,
        std::span<const int32_t> first_draft_tokens) noexcept
    {
        if (serial_tokens.size() < 2u || first_draft_tokens.empty())
            return std::nullopt;
        const size_t candidate_count = std::min(
            first_draft_tokens.size(), serial_tokens.size() - 1u);
        for (size_t step = 0u; step < candidate_count; ++step)
        {
            if (first_draft_tokens[step] != serial_tokens[step + 1u])
                continue;
            return MTPParityAcceptedDraftReference{
                .reference_step = step,
                .base_token = serial_tokens[step],
                .first_draft_token = serial_tokens[step + 1u],
            };
        }
        return std::nullopt;
    }

    /**
     * @brief Compare one grouped MTP response with a serial token prefix.
     *
     * @param serial_oracle Free-running serial Llaminar trajectory.
     * @param grouped_response Tokens returned by one production MTP transaction.
     * @return Exactness plus the first missing or unequal token boundary.
     *
     * An empty grouped response is invalid. A shorter serial oracle fails at
     * its first missing token rather than silently comparing a truncated
     * prefix. No Hugging Face token ids enter this contract; their role is the
     * separate tolerant logits/checkpoint comparison.
     */
    [[nodiscard]] constexpr MTPParityTokenComparison
    compareMTPGroupedTokensToSerialOracle(
        std::span<const int32_t> serial_oracle,
        std::span<const int32_t> grouped_response) noexcept
    {
        MTPParityTokenComparison result;
        if (grouped_response.empty())
        {
            result.mismatch_index = 0;
            return result;
        }

        const size_t comparable =
            serial_oracle.size() < grouped_response.size()
                ? serial_oracle.size()
                : grouped_response.size();
        for (size_t index = 0; index < comparable; ++index)
        {
            ++result.compared_tokens;
            if (serial_oracle[index] == grouped_response[index])
                continue;
            result.mismatch_index = index;
            result.serial_token = serial_oracle[index];
            result.grouped_token = grouped_response[index];
            return result;
        }

        if (serial_oracle.size() < grouped_response.size())
        {
            result.mismatch_index = serial_oracle.size();
            result.grouped_token = grouped_response[serial_oracle.size()];
            return result;
        }

        result.exact = true;
        return result;
    }

    /**
     * @brief Classify whether one decodeStep executed a speculative transaction.
     *
     * A terminal response-budget row may lawfully execute ordinary decode with
     * no draft. A real speculative observation must advance both counters;
     * advancing only one, or regressing either, is inconsistent evidence and
     * must fail closed. One device-resident decodeStep may retire multiple
     * transactions, so the two positive deltas intentionally need not match.
     */
    [[nodiscard]] constexpr MTPParityTransactionActivity
    classifyMTPParityTransactionActivity(
        MTPParityTransactionCounters before,
        MTPParityTransactionCounters after) noexcept
    {
        if (after.draft_steps < before.draft_steps ||
            after.verifier_runs < before.verifier_runs)
        {
            return MTPParityTransactionActivity::Inconsistent;
        }
        const bool drafted = after.draft_steps > before.draft_steps;
        const bool verified = after.verifier_runs > before.verifier_runs;
        if (drafted != verified)
            return MTPParityTransactionActivity::Inconsistent;
        return drafted ? MTPParityTransactionActivity::Speculative
                       : MTPParityTransactionActivity::None;
    }

    /** @brief Return the aggregate draft-token count retired by decodeStep(). */
    [[nodiscard]] constexpr uint64_t mtpParityAttemptedDraftTokenCount(
        MTPParityTransactionCounters before,
        MTPParityTransactionCounters after) noexcept
    {
        return after.draft_steps >= before.draft_steps
                   ? after.draft_steps - before.draft_steps
                   : 0;
    }

    /** @brief Return the number of verifier transactions retired by decodeStep(). */
    [[nodiscard]] constexpr uint64_t mtpParityExecutedTransactionCount(
        MTPParityTransactionCounters before,
        MTPParityTransactionCounters after) noexcept
    {
        return after.verifier_runs >= before.verifier_runs
                   ? after.verifier_runs - before.verifier_runs
                   : 0;
    }

    /**
     * @brief Return the smallest public budget that leaves one depth window unclipped.
     *
     * Draft depth @p draft_depth executes one condition row plus that many
     * speculative rows. A finite public budget equal to those verifier rows
     * must leave its final visible token pending for the next serial boundary,
     * so production shortens the live-state publication by one row and marks
     * the observation budget-limited. One additional response slot makes the
     * complete verifier unable to exhaust the public budget and therefore
     * permits the device depth controller to evaluate its full-width window.
     *
     * @param draft_depth Positive number of speculative comparison rows.
     * @return `draft_depth + 2`, or zero for invalid/overflowing input.
     */
    [[nodiscard]] constexpr int mtpParityFullWidthPolicyWitnessBudget(
        int draft_depth) noexcept
    {
        return draft_depth > 0 &&
                       draft_depth <= std::numeric_limits<int>::max() - 2
                   ? draft_depth + 2
                   : 0;
    }

    /**
     * @brief Map the live production condition position to an HF sidecar step.
     *
     * Reference sidecar step N consumes the condition token at logical
     * position `prompt_token_count + N`. The live position is the only exact
     * authority for that boundary. Response count is insufficient because a
     * rejected speculative token is already visible to the caller while it
     * remains the pending condition row for the next transaction. Fully
     * accepted and correction-pending transactions therefore can have the
     * same response count but different live condition positions.
     *
     * CPU probes obtain this value from host-owned sequence state; GPU probes
     * obtain it from the device-resident logical-state mailbox. The mapping
     * consequently preserves the engine's ownership boundary and never
     * reconstructs execution state from returned token shadows.
     *
     * @param live_condition_position Authoritative next-condition logical
     *        position reported immediately before the observed transaction.
     * @param prompt_token_count Number of tokens committed by prefill.
     * @return Exact HF recursive-sidecar decode step, or -1 when the inputs do
     *         not describe a position at or after the prefill boundary.
     */
    [[nodiscard]] constexpr int mtpParityReferenceStepForConditionPosition(
        int live_condition_position,
        int prompt_token_count) noexcept
    {
        return prompt_token_count >= 0 &&
                       live_condition_position >= prompt_token_count
                   ? live_condition_position - prompt_token_count
                   : -1;
    }

    /**
     * @brief Return the exact snapshot qualifier emitted for a sidecar role.
     *
     * The returned value includes its terminal underscore so concatenating a
     * relative stage such as `MTP0_EMBEDDING` produces the canonical key.
     */
    [[nodiscard]] constexpr std::string_view mtpParityCheckpointContextPrefix(
        MTPParityCheckpointContext context) noexcept
    {
        switch (context)
        {
        case MTPParityCheckpointContext::HostConditionTokenLivePosition:
            return "MTP_DECODE_SIDECAR_";
        case MTPParityCheckpointContext::HostChainedDraftLivePosition:
            return "MTP_DECODE_SIDECAR_CHAIN_";
        case MTPParityCheckpointContext::DeviceTargetTokenLivePosition:
            return "MTP_DECODE_SIDECAR_DEVICE_TARGET_TOKEN_LIVE_POSITION_";
        case MTPParityCheckpointContext::DeviceResidentLogicalState:
            return "MTP_DECODE_SIDECAR_RESIDENT_LOGICAL_STATE_";
        case MTPParityCheckpointContext::DeviceChainedTokenLivePosition:
            return "MTP_DECODE_SIDECAR_CHAIN_DEVICE_TOKEN_LIVE_POSITION_";
        }
        return {};
    }

    /**
     * @brief Extract the exact qualifier from a context-qualified model stage.
     *
     * A valid model checkpoint has an `MTP<depth>_<stage>` tail.  Operational
     * snapshots such as a live-position scalar deliberately return null even
     * when their names begin with the generic sidecar prefix.
     */
    [[nodiscard]] inline std::optional<std::string_view>
    mtpParityModelCheckpointContextPrefix(std::string_view key) noexcept
    {
        constexpr std::string_view sidecar_prefix =
            "MTP_DECODE_SIDECAR_";
        if (!key.starts_with(sidecar_prefix))
            return std::nullopt;

        size_t search_from = sidecar_prefix.size() - 1;
        while (true)
        {
            const size_t marker = key.find("_MTP", search_from);
            if (marker == std::string_view::npos)
                return std::nullopt;

            size_t cursor = marker + std::string_view("_MTP").size();
            const size_t first_digit = cursor;
            while (cursor < key.size() &&
                   key[cursor] >= '0' && key[cursor] <= '9')
            {
                ++cursor;
            }
            if (cursor > first_digit && cursor + 1 < key.size() &&
                key[cursor] == '_')
            {
                return key.substr(0, marker + 1);
            }
            search_from = marker + 1;
        }
    }

    /** @brief Return whether a key is a context-qualified MTP model stage. */
    [[nodiscard]] inline bool isMTPParityModelCheckpointSnapshotKey(
        std::string_view key) noexcept
    {
        return mtpParityModelCheckpointContextPrefix(key).has_value();
    }
} // namespace llaminar2::test::parity
