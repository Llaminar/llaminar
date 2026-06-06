#include "MTPStateTransaction.h"

#include <algorithm>
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

} // namespace llaminar2
