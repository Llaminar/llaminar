#pragma once

#include "execution/prefix_cache/PrefixStorageBackend.h"

#include <cstdint>

namespace llaminar2
{

    struct PrefixStateBlock
    {
        PrefixBlockHandle handle;
        int cached_tokens = 0;
        int block_index = 0;
        uint64_t last_access_tick = 0;
        uint32_t ref_count = 0;
    };

} // namespace llaminar2
