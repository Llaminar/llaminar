#pragma once

#include "execution/prefix_cache/PrefixStorageBackend.h"

#include <cstdint>
#include <string>
#include <vector>

namespace llaminar2
{

    struct PrefixLookupResult
    {
        bool supported = false;
        bool cache_enabled = false;
        int cached_tokens = 0;
        int block_size = 0;
        uint64_t fingerprint_key = 0;
        uint64_t placement_epoch = 0;
        bool requires_terminal_hidden = true;
        bool requires_terminal_logits = true;
        bool has_terminal_hidden = false;
        bool has_terminal_logits = false;
        std::string bypass_reason;
        std::vector<PrefixBlockHandle> blocks;

        bool hit() const { return supported && cached_tokens > 0 && !blocks.empty(); }
        PrefixLookupResult clampedTo(int token_count) const;
    };

    struct PrefixStateSnapshot
    {
        bool valid = false;
        int cached_tokens = 0;
        std::vector<PrefixBlockHandle> blocks;
        std::vector<PrefixBlockHandle> mtp_blocks;
        std::vector<PrefixStateSnapshot> participant_snapshots;
    };

} // namespace llaminar2
