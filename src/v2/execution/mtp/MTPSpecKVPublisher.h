#pragma once

#include "MTPSpecStateContract.h"

#include <string>
#include <vector>

namespace llaminar2
{
    class IKVCache;

    struct MTPSpecKVPublicationResult
    {
        bool ok = false;
        std::string error;

        int request_id = -1;
        int seq_idx = 0;
        int target_cached_tokens = 0;
        int main_truncated_tokens = 0;
        std::vector<int> mtp_truncated_tokens;
    };

    MTPSpecKVPublicationResult publishAcceptedMTPSpecKVState(
        const MTPSpecStepPlan &plan,
        IKVCache &main_cache,
        const std::vector<IKVCache *> &mtp_caches,
        int seq_idx,
        void *stream);

} // namespace llaminar2
