#include "execution/prefix_cache/PrefixStateSnapshot.h"

#include <algorithm>

namespace llaminar2
{

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

        const int wanted_blocks = result.cached_tokens / block_size;
        const int copy_blocks = std::min<int>(wanted_blocks, blocks.size());
        result.blocks.insert(result.blocks.end(), blocks.begin(), blocks.begin() + copy_blocks);
        result.cached_tokens = copy_blocks * block_size;

        if (!result.blocks.empty() && result.cached_tokens == cached_tokens)
        {
            result.has_terminal_hidden = has_terminal_hidden;
            result.has_terminal_logits = has_terminal_logits;
        }

        return result;
    }

} // namespace llaminar2
