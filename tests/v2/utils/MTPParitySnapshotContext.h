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

#include <cstdint>
#include <optional>
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
