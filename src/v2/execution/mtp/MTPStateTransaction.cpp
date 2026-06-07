#include "MTPStateTransaction.h"

#include <algorithm>
#include <sstream>
#include <utility>

namespace llaminar2
{

    MTPStateValidationResult MTPStateValidationResult::success()
    {
        return {true, {}};
    }

    MTPStateValidationResult MTPStateValidationResult::failure(std::string reason)
    {
        return {false, std::move(reason)};
    }

    int expectedShiftedMTPTokens(int logical_tokens)
    {
        return std::max(0, logical_tokens - 1);
    }

    MTPStateValidationResult validateCommittedMTPDecodeState(
        const MTPDecodeStateStamp &state,
        const MTPCommitValidationOptions &options)
    {
        if (!state.valid)
            return MTPStateValidationResult::failure("state is invalid");
        if (state.logical_tokens < 0 ||
            state.main_kv_tokens < 0 ||
            state.shifted_mtp_kv_tokens < 0 ||
            state.position < 0)
        {
            return MTPStateValidationResult::failure("state contains negative token or position counts");
        }
        if (state.main_kv_tokens != state.logical_tokens)
        {
            return MTPStateValidationResult::failure("main KV token count does not match logical token count");
        }
        if (state.position != state.logical_tokens)
        {
            return MTPStateValidationResult::failure("decode position does not match logical token count");
        }
        if (state.shifted_mtp_kv_tokens != expectedShiftedMTPTokens(state.logical_tokens))
        {
            return MTPStateValidationResult::failure("shifted MTP KV token count does not match logical token count");
        }
        if (options.require_decode_equivalent_source && !state.decodeEquivalent())
        {
            return MTPStateValidationResult::failure(
                std::string("state provenance is not decode-equivalent: ") +
                toString(state.provenance));
        }
        if (options.require_terminal_hidden && !state.has_terminal_hidden)
            return MTPStateValidationResult::failure("terminal hidden is missing");
        if (options.require_terminal_logits && !state.has_terminal_logits)
            return MTPStateValidationResult::failure("terminal logits are missing");
        if (options.require_ready_token && !state.has_ready_token)
            return MTPStateValidationResult::failure("ready token is missing");
        return MTPStateValidationResult::success();
    }

    MTPStateValidationResult validateAtomicMTPCommit(
        const MTPDecodeStateStamp &base,
        const MTPDecodeStateStamp &committed,
        int emitted_tokens,
        PrefixStateProvenance verifier_source,
        const MTPCommitValidationOptions &options)
    {
        if (emitted_tokens <= 0)
            return MTPStateValidationResult::failure("atomic MTP commit emitted no tokens");
        auto base_result = validateCommittedMTPDecodeState(base, options);
        if (!base_result)
            return MTPStateValidationResult::failure("base state failed validation: " + base_result.reason);
        auto committed_result = validateCommittedMTPDecodeState(committed, options);
        if (!committed_result)
            return MTPStateValidationResult::failure("committed state failed validation: " + committed_result.reason);
        if (options.require_decode_equivalent_source && !isDecodeEquivalent(verifier_source))
        {
            return MTPStateValidationResult::failure(
                std::string("verifier source is not decode-equivalent: ") +
                toString(verifier_source));
        }
        if (committed.logical_tokens != base.logical_tokens + emitted_tokens)
        {
            return MTPStateValidationResult::failure("committed logical token count does not equal base plus emitted tokens");
        }
        if (committed.main_kv_tokens < base.main_kv_tokens ||
            committed.shifted_mtp_kv_tokens < base.shifted_mtp_kv_tokens)
        {
            return MTPStateValidationResult::failure("committed state moved KV token counts backwards");
        }
        return MTPStateValidationResult::success();
    }

    MTPStateValidationResult compareMTPRuntimeStateSnapshots(
        const PrefixRuntimeStateSnapshot &oracle,
        const PrefixRuntimeStateSnapshot &candidate)
    {
        auto mismatch = [](std::string reason)
        {
            return MTPStateValidationResult::failure(std::move(reason));
        };

        if (oracle.initialized != candidate.initialized)
            return mismatch("initialized flag mismatch");
        if (!oracle.initialized)
            return MTPStateValidationResult::success();
        if (oracle.current_position != candidate.current_position)
            return mismatch("current position mismatch");
        if (oracle.positions != candidate.positions)
            return mismatch("per-sequence position vector mismatch");
        if (oracle.sequence_lengths != candidate.sequence_lengths)
            return mismatch("per-sequence length vector mismatch");
        if (oracle.totalCachedTokens() != candidate.totalCachedTokens())
            return mismatch("main KV cached token count mismatch");
        if (oracle.totalMTPCachedTokens() != candidate.totalMTPCachedTokens())
            return mismatch("shifted MTP cached token count mismatch");
        if (oracle.has_hidden != candidate.has_hidden)
            return mismatch("terminal hidden availability mismatch");
        if (oracle.has_logits != candidate.has_logits)
            return mismatch("terminal logits availability mismatch");
        if (oracle.gdn_layers.size() != candidate.gdn_layers.size())
            return mismatch("GDN layer count mismatch");

        for (size_t i = 0; i < oracle.gdn_layers.size(); ++i)
        {
            const PrefixGDNLayerProbe &lhs = oracle.gdn_layers[i];
            const PrefixGDNLayerProbe &rhs = candidate.gdn_layers[i];
            if (lhs.global_layer != rhs.global_layer)
                return mismatch("GDN layer id mismatch");
            if (lhs.recurrence_values != rhs.recurrence_values)
                return mismatch("GDN recurrence value count mismatch");
            if (lhs.conv_values != rhs.conv_values)
                return mismatch("GDN short-conv value count mismatch");
            if (lhs.recurrence_hash != rhs.recurrence_hash)
            {
                std::ostringstream msg;
                msg << "GDN recurrence hash mismatch at layer "
                    << lhs.global_layer;
                return mismatch(msg.str());
            }
            if (lhs.conv_hash != rhs.conv_hash)
            {
                std::ostringstream msg;
                msg << "GDN short-conv hash mismatch at layer "
                    << lhs.global_layer;
                return mismatch(msg.str());
            }
            if (lhs.recurrence_all_zero != rhs.recurrence_all_zero)
                return mismatch("GDN recurrence zero-state flag mismatch");
            if (lhs.conv_all_zero != rhs.conv_all_zero)
                return mismatch("GDN short-conv zero-state flag mismatch");
        }

        return MTPStateValidationResult::success();
    }

} // namespace llaminar2
