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

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>

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
