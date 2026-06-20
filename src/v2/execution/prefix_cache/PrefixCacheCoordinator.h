#pragma once

#include "backends/DeviceId.h"
#include "execution/prefix_cache/PrefixStateSnapshot.h"

#include <mpi.h>
#include <string>
#include <vector>

namespace llaminar2
{
    struct PrefixParticipantLookup
    {
        std::string domain_id;
        int participant_id = -1;
        DeviceId device = DeviceId::cpu();
        uint64_t placement_epoch = 0;
        uint64_t fingerprint_key = 0;
        bool fingerprint_must_match = true;
        bool supported = false;
        bool cache_enabled = false;
        bool hit = false;
        int matched_tokens = 0;
        int matched_blocks = 0;
        bool requires_terminal_logits = true;
        bool requires_terminal_hidden = true;
        bool has_terminal_logits = false;
        bool has_terminal_hidden = false;
        std::string bypass_reason;
    };

    struct PrefixCoordinationResult
    {
        std::string domain_id;
        uint64_t placement_epoch = 0;
        uint64_t fingerprint_key = 0;
        bool supported = false;
        bool cache_enabled = false;
        int common_matched_tokens = 0;
        int common_matched_blocks = 0;
        bool common_terminal_logits_required = false;
        bool common_terminal_hidden_required = false;
        bool common_terminal_logits = false;
        bool common_terminal_hidden = false;
        std::string clamp_reason;
        std::vector<PrefixParticipantLookup> participants;

        bool hit() const
        {
            return supported && cache_enabled && common_matched_tokens > 0;
        }
    };

    class IPrefixCollectiveCoordinator
    {
    public:
        virtual ~IPrefixCollectiveCoordinator() = default;

        virtual bool allMinInt(int local_value, int *global_value) = 0;
        virtual bool allMinUInt64(uint64_t local_value, uint64_t *global_value) = 0;
        virtual bool allMaxUInt64(uint64_t local_value, uint64_t *global_value) = 0;
        virtual bool allAndBool(bool local_value, bool *global_value) = 0;
        virtual bool allOrBool(bool local_value, bool *global_value) = 0;
    };

    class MPIPrefixCollectiveCoordinator : public IPrefixCollectiveCoordinator
    {
    public:
        explicit MPIPrefixCollectiveCoordinator(MPI_Comm communicator);

        bool allMinInt(int local_value, int *global_value) override;
        bool allMinUInt64(uint64_t local_value, uint64_t *global_value) override;
        bool allMaxUInt64(uint64_t local_value, uint64_t *global_value) override;
        bool allAndBool(bool local_value, bool *global_value) override;
        bool allOrBool(bool local_value, bool *global_value) override;

    private:
        MPI_Comm communicator_ = MPI_COMM_NULL;
    };

    PrefixParticipantLookup makePrefixParticipantLookup(
        int participant_id,
        DeviceId device,
        const PrefixLookupResult &hit,
        std::string domain_id = {},
        uint64_t placement_epoch = 0);

    PrefixCoordinationResult coordinatePrefixLookups(
        std::vector<PrefixParticipantLookup> participants,
        IPrefixCollectiveCoordinator *domain_coordinator = nullptr);

    PrefixLookupResult makePrefixLookupResult(
        const PrefixCoordinationResult &coordination,
        int block_size);

} // namespace llaminar2
