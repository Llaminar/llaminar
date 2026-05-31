#include "execution/prefix_cache/PrefixCacheCoordinator.h"

#include <algorithm>
#include <limits>
#include <utility>

namespace llaminar2
{
    namespace
    {
        int nonNegative(int value)
        {
            return std::max(0, value);
        }

        std::string firstParticipantReason(const std::vector<PrefixParticipantLookup> &participants)
        {
            for (const auto &participant : participants)
            {
                if (!participant.supported)
                {
                    if (!participant.bypass_reason.empty())
                        return participant.bypass_reason;
                    return "participant " + std::to_string(participant.participant_id) +
                           " does not support prefix cache";
                }
            }
            return {};
        }

        std::string commonDomainId(const std::vector<PrefixParticipantLookup> &participants)
        {
            std::string domain_id;
            bool saw_domain = false;
            for (const auto &participant : participants)
            {
                if (participant.domain_id.empty())
                    continue;
                if (!saw_domain)
                {
                    domain_id = participant.domain_id;
                    saw_domain = true;
                    continue;
                }
                if (domain_id != participant.domain_id)
                    return "mixed";
            }
            return domain_id;
        }
    } // namespace

    MPIPrefixCollectiveCoordinator::MPIPrefixCollectiveCoordinator(MPI_Comm communicator)
        : communicator_(communicator)
    {
    }

    bool MPIPrefixCollectiveCoordinator::allMinInt(int local_value, int *global_value)
    {
        if (!global_value || communicator_ == MPI_COMM_NULL)
            return false;
        return MPI_Allreduce(&local_value, global_value, 1, MPI_INT, MPI_MIN, communicator_) == MPI_SUCCESS;
    }

    bool MPIPrefixCollectiveCoordinator::allMinUInt64(uint64_t local_value, uint64_t *global_value)
    {
        if (!global_value || communicator_ == MPI_COMM_NULL)
            return false;
        return MPI_Allreduce(&local_value, global_value, 1, MPI_UINT64_T, MPI_MIN, communicator_) == MPI_SUCCESS;
    }

    bool MPIPrefixCollectiveCoordinator::allMaxUInt64(uint64_t local_value, uint64_t *global_value)
    {
        if (!global_value || communicator_ == MPI_COMM_NULL)
            return false;
        return MPI_Allreduce(&local_value, global_value, 1, MPI_UINT64_T, MPI_MAX, communicator_) == MPI_SUCCESS;
    }

    bool MPIPrefixCollectiveCoordinator::allAndBool(bool local_value, bool *global_value)
    {
        if (!global_value || communicator_ == MPI_COMM_NULL)
            return false;
        const int local_int = local_value ? 1 : 0;
        int global_int = 0;
        if (MPI_Allreduce(&local_int, &global_int, 1, MPI_INT, MPI_MIN, communicator_) != MPI_SUCCESS)
            return false;
        *global_value = global_int != 0;
        return true;
    }

    bool MPIPrefixCollectiveCoordinator::allOrBool(bool local_value, bool *global_value)
    {
        if (!global_value || communicator_ == MPI_COMM_NULL)
            return false;
        const int local_int = local_value ? 1 : 0;
        int global_int = 0;
        if (MPI_Allreduce(&local_int, &global_int, 1, MPI_INT, MPI_MAX, communicator_) != MPI_SUCCESS)
            return false;
        *global_value = global_int != 0;
        return true;
    }

    PrefixParticipantLookup makePrefixParticipantLookup(
        int participant_id,
        DeviceId device,
        const PrefixLookupResult &hit,
        std::string domain_id,
        uint64_t placement_epoch)
    {
        PrefixParticipantLookup participant;
        participant.domain_id = std::move(domain_id);
        participant.participant_id = participant_id;
        participant.device = device;
        participant.placement_epoch = placement_epoch;
        participant.fingerprint_key = hit.fingerprint_key;
        participant.supported = hit.supported;
        participant.cache_enabled = hit.cache_enabled;
        participant.hit = hit.cached_tokens > 0;
        participant.matched_tokens = nonNegative(hit.cached_tokens);
        participant.matched_blocks = !hit.blocks.empty()
                                         ? static_cast<int>(hit.blocks.size())
                                         : (hit.block_size > 0 ? participant.matched_tokens / hit.block_size : 0);
        participant.has_terminal_logits = hit.has_terminal_logits;
        participant.has_terminal_hidden = hit.has_terminal_hidden;
        participant.bypass_reason = hit.bypass_reason;
        return participant;
    }

    PrefixCoordinationResult coordinatePrefixLookups(
        std::vector<PrefixParticipantLookup> participants,
        IPrefixCollectiveCoordinator *domain_coordinator)
    {
        PrefixCoordinationResult result;
        result.participants = std::move(participants);
        result.domain_id = commonDomainId(result.participants);

        if (result.participants.empty())
        {
            result.clamp_reason = "no prefix participants";
            return result;
        }

        bool local_any_cache_enabled = false;
        bool local_all_supported = true;
        bool local_all_terminal_logits = true;
        bool local_all_terminal_hidden = true;
        int local_min_tokens = std::numeric_limits<int>::max();
        int local_min_blocks = std::numeric_limits<int>::max();
        int local_max_tokens = 0;
        uint64_t local_min_fingerprint = std::numeric_limits<uint64_t>::max();
        uint64_t local_max_fingerprint = 0;
        uint64_t local_placement_epoch = 0;

        for (const auto &participant : result.participants)
        {
            local_any_cache_enabled = local_any_cache_enabled || participant.cache_enabled;
            local_all_supported = local_all_supported && participant.supported;

            const int participant_tokens = participant.supported ? nonNegative(participant.matched_tokens) : 0;
            const int participant_blocks = participant.supported ? nonNegative(participant.matched_blocks) : 0;
            local_min_tokens = std::min(local_min_tokens, participant_tokens);
            local_min_blocks = std::min(local_min_blocks, participant_blocks);
            local_max_tokens = std::max(local_max_tokens, participant_tokens);
            local_placement_epoch = std::max(local_placement_epoch, participant.placement_epoch);
            if (participant_tokens > 0 && participant.fingerprint_key != 0)
            {
                local_min_fingerprint = std::min(local_min_fingerprint, participant.fingerprint_key);
                local_max_fingerprint = std::max(local_max_fingerprint, participant.fingerprint_key);
            }

            local_all_terminal_logits = local_all_terminal_logits && participant.has_terminal_logits;
            local_all_terminal_hidden = local_all_terminal_hidden && participant.has_terminal_hidden;
        }

        if (local_min_tokens == std::numeric_limits<int>::max())
            local_min_tokens = 0;
        if (local_min_blocks == std::numeric_limits<int>::max())
            local_min_blocks = 0;
        if (local_min_fingerprint == std::numeric_limits<uint64_t>::max())
            local_min_fingerprint = 0;

        int global_min_tokens = local_min_tokens;
        int global_min_blocks = local_min_blocks;
        uint64_t global_min_fingerprint = local_min_fingerprint;
        uint64_t global_max_fingerprint = local_max_fingerprint;
        uint64_t global_placement_epoch = local_placement_epoch;
        bool global_any_cache_enabled = local_any_cache_enabled;
        bool global_all_supported = local_all_supported;
        bool global_all_terminal_logits = local_all_terminal_logits;
        bool global_all_terminal_hidden = local_all_terminal_hidden;

        if (domain_coordinator)
        {
            if (!domain_coordinator->allMinInt(local_min_tokens, &global_min_tokens) ||
                !domain_coordinator->allMinInt(local_min_blocks, &global_min_blocks) ||
                !domain_coordinator->allMinUInt64(local_min_fingerprint, &global_min_fingerprint) ||
                !domain_coordinator->allMaxUInt64(local_max_fingerprint, &global_max_fingerprint) ||
                !domain_coordinator->allMaxUInt64(local_placement_epoch, &global_placement_epoch) ||
                !domain_coordinator->allOrBool(local_any_cache_enabled, &global_any_cache_enabled) ||
                !domain_coordinator->allAndBool(local_all_supported, &global_all_supported) ||
                !domain_coordinator->allAndBool(local_all_terminal_logits, &global_all_terminal_logits) ||
                !domain_coordinator->allAndBool(local_all_terminal_hidden, &global_all_terminal_hidden))
            {
                result.cache_enabled = global_any_cache_enabled;
                result.supported = false;
                result.clamp_reason = "prefix coordination collective failed";
                return result;
            }
        }

        result.cache_enabled = global_any_cache_enabled;
        result.supported = global_all_supported;
        result.placement_epoch = global_placement_epoch;
        const bool fingerprint_mismatch =
            global_min_tokens > 0 &&
            global_min_fingerprint != 0 &&
            global_max_fingerprint != 0 &&
            global_min_fingerprint != global_max_fingerprint;
        result.supported = result.supported && !fingerprint_mismatch;
        result.common_matched_tokens = result.supported ? nonNegative(global_min_tokens) : 0;
        result.common_matched_blocks = result.supported ? nonNegative(global_min_blocks) : 0;
        result.fingerprint_key =
            result.common_matched_tokens > 0 &&
                    global_min_fingerprint == global_max_fingerprint
                ? global_min_fingerprint
                : 0;
        result.common_terminal_logits =
            result.common_matched_tokens > 0 && global_all_terminal_logits;
        result.common_terminal_hidden =
            result.common_matched_tokens > 0 && global_all_terminal_hidden;

        if (fingerprint_mismatch)
        {
            result.clamp_reason = "prefix fingerprint mismatch across participants";
        }
        else if (!result.supported)
        {
            result.clamp_reason = firstParticipantReason(result.participants);
            if (result.clamp_reason.empty())
                result.clamp_reason = "at least one prefix participant is unsupported";
        }
        else if (result.common_matched_tokens < local_max_tokens)
        {
            result.clamp_reason = "clamped to common prefix across participants";
        }
        else if (result.common_matched_tokens > 0 &&
                 (!result.common_terminal_logits || !result.common_terminal_hidden))
        {
            result.clamp_reason = "terminal state missing on at least one participant";
        }

        return result;
    }

    PrefixLookupResult makePrefixLookupResult(
        const PrefixCoordinationResult &coordination,
        int block_size)
    {
        PrefixLookupResult result;
        result.supported = coordination.supported;
        result.cache_enabled = coordination.cache_enabled;
        result.block_size = block_size;
        result.fingerprint_key = coordination.fingerprint_key;

        int restorable_tokens = nonNegative(coordination.common_matched_tokens);
        if (block_size > 0)
        {
            const int block_tokens =
                nonNegative(coordination.common_matched_blocks) * block_size;
            restorable_tokens = std::min(restorable_tokens, block_tokens);
        }

        result.cached_tokens = restorable_tokens;
        const bool preserved_terminal_token =
            restorable_tokens > 0 &&
            restorable_tokens == coordination.common_matched_tokens;
        result.has_terminal_logits =
            preserved_terminal_token && coordination.common_terminal_logits;
        result.has_terminal_hidden =
            preserved_terminal_token && coordination.common_terminal_hidden;
        if (!coordination.supported)
            result.bypass_reason = coordination.clamp_reason;
        return result;
    }

} // namespace llaminar2
