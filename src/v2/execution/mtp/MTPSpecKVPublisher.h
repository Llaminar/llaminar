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

    /**
     * @brief Compute the publish target for one shifted MTP KV depth.
     *
     * The main verifier cache advances by `accepted_count` rows.  Shifted MTP
     * cache depth zero is ordinarily one row behind the main model, depth one
     * is two rows behind, and so on.  Some sidecar paths have already appended
     * the verifier row-zero shifted KV row before publication; paths that do
     * not set `reuse_initial_mtp_shifted_kv_row` have restored that sidecar row
     * away and must start one additional row behind before publishing only
     * verifier rows strictly beyond the row-zero boundary.
     */
    int computeMTPShiftedKVTargetCachedTokens(
        const MTPSpecStepPlan &plan,
        int mtp_depth);

} // namespace llaminar2
