#include "execution/prefix_cache/PrefixStateSnapshot.h"

#include <algorithm>

namespace llaminar2
{

    const char *toString(PrefixStateProvenance provenance)
    {
        switch (provenance)
        {
        case PrefixStateProvenance::Unknown:
            return "unknown";
        case PrefixStateProvenance::PayloadCheckpoint:
            return "payload_checkpoint";
        case PrefixStateProvenance::LogicalCheckpoint:
            return "logical_checkpoint";
        case PrefixStateProvenance::DecodeEquivalent:
            return "decode_equivalent";
        case PrefixStateProvenance::VerifierPrefillRows:
            return "verifier_prefill_rows";
        case PrefixStateProvenance::VerifierPrefillRowsDecodeEquivalent:
            return "verifier_prefill_rows_decode_equivalent";
        case PrefixStateProvenance::SidecarDraftOnly:
            return "sidecar_draft_only";
        default:
            return "unknown";
        }
    }

    bool isDecodeEquivalent(PrefixStateProvenance provenance)
    {
        switch (provenance)
        {
        case PrefixStateProvenance::PayloadCheckpoint:
        case PrefixStateProvenance::LogicalCheckpoint:
        case PrefixStateProvenance::DecodeEquivalent:
        case PrefixStateProvenance::VerifierPrefillRowsDecodeEquivalent:
            return true;
        case PrefixStateProvenance::Unknown:
        case PrefixStateProvenance::VerifierPrefillRows:
        case PrefixStateProvenance::SidecarDraftOnly:
            return false;
        default:
            return false;
        }
    }

    PrefixTerminalHarvestDisposition
    PrefixLookupResult::terminalHarvestDisposition(
        const PrefixCacheKey &terminal_key,
        int prompt_token_count) const
    {
        if (!supported || !cache_enabled || prompt_token_count <= 0 ||
            cached_tokens != prompt_token_count || blocks.empty() ||
            fingerprint_key != terminal_key.fingerprint)
        {
            return PrefixTerminalHarvestDisposition::ArchiveLiveState;
        }

        const PrefixBlockHandle &terminal = blocks.back();
        const bool exact_terminal =
            terminal.valid() && terminal.key == terminal_key &&
            terminal.key.token_start + terminal.key.token_count ==
                prompt_token_count;
        const bool terminal_logits_complete =
            !requires_terminal_logits ||
            (has_terminal_logits && terminal.has_terminal_logits);
        const bool terminal_hidden_complete =
            !requires_terminal_hidden ||
            (has_terminal_hidden && terminal.has_terminal_hidden);
        const bool hybrid_state_complete =
            !terminal.layout.includes_hybrid_state ||
            terminal.has_hybrid_state;

        return exact_terminal && terminal_logits_complete &&
                       terminal_hidden_complete && hybrid_state_complete
                   ? PrefixTerminalHarvestDisposition::ReuseAdmittedArchive
                   : PrefixTerminalHarvestDisposition::ArchiveLiveState;
    }

    PrefixLookupResult PrefixLookupResult::clampedTo(int token_count) const
    {
        PrefixLookupResult result = *this;
        result.cached_tokens = std::max(0, std::min(cached_tokens, token_count));
        result.blocks.clear();
        result.has_terminal_hidden = false;
        result.has_terminal_logits = false;

        if (result.cached_tokens == 0 || block_size <= 0)
        {
            return result;
        }
        if (blocks.empty())
        {
            result.has_terminal_hidden = result.cached_tokens == cached_tokens && has_terminal_hidden;
            result.has_terminal_logits = result.cached_tokens == cached_tokens && has_terminal_logits;
            return result;
        }

        for (const auto &block : blocks)
        {
            const int block_end = block.key.token_start + block.key.token_count;
            if (block.key.token_count <= 0 || block_end > result.cached_tokens)
            {
                break;
            }
            result.blocks.push_back(block);
        }

        result.cached_tokens = result.blocks.empty()
                                   ? 0
                                   : result.blocks.back().key.token_start +
                                         result.blocks.back().key.token_count;

        while (!result.blocks.empty() &&
               result.blocks.back().layout.hybrid_state_bytes > 0 &&
               !result.blocks.back().has_hybrid_state)
        {
            result.blocks.pop_back();
            result.cached_tokens = result.blocks.empty()
                                       ? 0
                                       : result.blocks.back().key.token_start +
                                             result.blocks.back().key.token_count;
        }

        if (!result.blocks.empty() && result.cached_tokens == cached_tokens)
        {
            result.has_terminal_hidden = has_terminal_hidden;
            result.has_terminal_logits = has_terminal_logits;
        }

        return result;
    }

} // namespace llaminar2
