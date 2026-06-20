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

    int computeMTPShiftedKVTargetCachedTokens(
        const MTPSpecStepPlan &plan,
        int mtp_depth)
    {
        if (mtp_depth < 0)
            return 0;

        const int shift = mtp_depth + 1;
        /*
         * Shifted MTP KV is a cache-shape invariant, not a sidecar-reuse
         * policy: once a transaction commits to main logical token count N,
         * depth d must expose N - (d + 1) rows.  Paths that cannot safely
         * publish the first shifted row must repair it before calling the
         * publisher, or fail hard.  Keeping the host mirror on this same rule
         * is especially important for device-resident publication, where the
         * GPU metadata kernel already publishes this target and the host mirror
         * is adopted later for planning and validation.
         */
        return std::max(0, plan.target_cached_tokens - shift);
    }

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
                computeMTPShiftedKVTargetCachedTokens(
                    plan,
                    static_cast<int>(depth));
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
                const int current_tokens =
                    cache->get_cached_tokens(cache->first_layer_index(), seq_idx);
                std::ostringstream msg;
                msg << "MTP spec KV publication failed truncating MTP KV depth "
                    << depth << " to " << shifted_tokens
                    << " tokens (current=" << current_tokens
                    << " base=" << plan.base_cached_tokens
                    << " accepted=" << plan.accepted_count
                    << " target=" << plan.target_cached_tokens
                    << " reuse_initial_shifted_row="
                    << (plan.reuse_initial_mtp_shifted_kv_row ? "true" : "false")
                    << ")";
                return kvPublicationFailure(plan, seq_idx, msg.str());
            }
            result.mtp_truncated_tokens.push_back(shifted_tokens);
        }

        return result;
    }

} // namespace llaminar2
