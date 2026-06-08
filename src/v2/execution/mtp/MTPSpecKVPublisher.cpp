#include "MTPSpecKVPublisher.h"

#include "../../kernels/IKVCache.h"

#include <algorithm>
#include <sstream>
#include <utility>

namespace llaminar2
{
    namespace
    {
        MTPSpecKVPublicationResult kvPublicationFailure(
            const MTPSpecStepPlan &plan,
            int seq_idx,
            std::string reason)
        {
            MTPSpecKVPublicationResult result;
            result.ok = false;
            result.error = std::move(reason);
            result.request_id = plan.request_id;
            result.seq_idx = seq_idx;
            result.target_cached_tokens = plan.target_cached_tokens;
            return result;
        }

        bool validatePlanShape(const MTPSpecStepPlan &plan, std::string *error)
        {
            if (plan.draft_count < 0 || plan.target_rows != plan.draft_count + 1)
            {
                if (error)
                    *error = "MTP spec KV publication received invalid draft/target row shape";
                return false;
            }
            if (plan.accepted_count < 0 || plan.accepted_count > plan.draft_count)
            {
                if (error)
                    *error = "MTP spec KV publication accepted count is outside the draft prefix";
                return false;
            }
            if (plan.base_cached_tokens < 0)
            {
                if (error)
                    *error = "MTP spec KV publication base cached-token count is negative";
                return false;
            }
            if (plan.target_cached_tokens != plan.base_cached_tokens + plan.accepted_count)
            {
                if (error)
                {
                    std::ostringstream msg;
                    msg << "MTP spec KV publication target cached-token count drifted from plan: target="
                        << plan.target_cached_tokens
                        << " expected=" << (plan.base_cached_tokens + plan.accepted_count);
                    *error = msg.str();
                }
                return false;
            }
            return true;
        }
    } // namespace

    MTPSpecKVPublicationResult publishAcceptedMTPSpecKVState(
        const MTPSpecStepPlan &plan,
        IKVCache &main_cache,
        const std::vector<IKVCache *> &mtp_caches,
        int seq_idx,
        void *stream)
    {
        std::string validation_error;
        if (!validatePlanShape(plan, &validation_error))
            return kvPublicationFailure(plan, seq_idx, std::move(validation_error));

        if (seq_idx < 0)
            return kvPublicationFailure(plan, seq_idx, "MTP spec KV publication received negative sequence index");
        if (plan.target_cached_tokens > main_cache.max_seq_len())
        {
            std::ostringstream msg;
            msg << "MTP spec KV publication target cached-token count exceeds main KV capacity: target="
                << plan.target_cached_tokens << " max=" << main_cache.max_seq_len();
            return kvPublicationFailure(plan, seq_idx, msg.str());
        }

        MTPSpecKVPublicationResult result;
        result.ok = true;
        result.request_id = plan.request_id;
        result.seq_idx = seq_idx;
        result.target_cached_tokens = plan.target_cached_tokens;

        if (!main_cache.truncateSequence(seq_idx, plan.target_cached_tokens, stream))
        {
            std::ostringstream msg;
            msg << "MTP spec KV publication failed truncating main KV to "
                << plan.target_cached_tokens << " tokens";
            return kvPublicationFailure(plan, seq_idx, msg.str());
        }
        result.main_truncated_tokens = plan.target_cached_tokens;

        result.mtp_truncated_tokens.reserve(mtp_caches.size());
        for (size_t depth = 0; depth < mtp_caches.size(); ++depth)
        {
            IKVCache *cache = mtp_caches[depth];
            if (cache == nullptr)
            {
                std::ostringstream msg;
                msg << "MTP spec KV publication received null MTP KV cache at depth " << depth;
                return kvPublicationFailure(plan, seq_idx, msg.str());
            }

            const int shifted_tokens =
                std::max(0, plan.target_cached_tokens - static_cast<int>(depth) - 1);
            if (shifted_tokens > cache->max_seq_len())
            {
                std::ostringstream msg;
                msg << "MTP spec KV publication shifted cache target exceeds capacity at depth "
                    << depth << ": target=" << shifted_tokens
                    << " max=" << cache->max_seq_len();
                return kvPublicationFailure(plan, seq_idx, msg.str());
            }
            if (!cache->truncateSequence(seq_idx, shifted_tokens, stream))
            {
                std::ostringstream msg;
                msg << "MTP spec KV publication failed truncating MTP KV depth "
                    << depth << " to " << shifted_tokens << " tokens";
                return kvPublicationFailure(plan, seq_idx, msg.str());
            }
            result.mtp_truncated_tokens.push_back(shifted_tokens);
        }

        return result;
    }

} // namespace llaminar2
