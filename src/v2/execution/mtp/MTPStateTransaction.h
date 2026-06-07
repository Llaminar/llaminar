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

    int expectedShiftedMTPTokens(int logical_tokens);

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
        const PrefixRuntimeStateSnapshot &candidate);

} // namespace llaminar2
