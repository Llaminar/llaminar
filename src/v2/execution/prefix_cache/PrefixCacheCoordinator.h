/**
 * @file PrefixCacheCoordinator.h
 * @brief Typed common-prefix coordination across partitioned inference participants.
 *
 * Each participant validates its own cache fingerprint before reporting a hit.
 * Coordination then clamps all reports to one restorable token boundary and
 * verifies terminal-state availability. Replicated payloads additionally
 * require one identical fingerprint; TP/PP payload slices deliberately do not.
 */

#pragma once

#include "backends/DeviceId.h"
#include "execution/prefix_cache/PrefixStateSnapshot.h"

#include <cstdint>
#include <mpi.h>
#include <string>
#include <vector>

namespace llaminar2
{
    /** Fingerprint relationship between coordinated prefix payload owners. */
    enum class PrefixFingerprintCoordinationPolicy : std::uint8_t
    {
        /** Every participant is a replica of the same payload. */
        RequireIdentical,
        /** Every participant locally validated a distinct TP/PP payload slice. */
        ValidateParticipantLocally,
    };

    /** One participant's immutable result at a common-prefix boundary. */
    struct PrefixParticipantLookup
    {
        std::string domain_id;
        int participant_id = -1;
        DeviceId device = DeviceId::cpu();
        uint64_t placement_epoch = 0;
        uint64_t fingerprint_key = 0;
        PrefixFingerprintCoordinationPolicy fingerprint_policy =
            PrefixFingerprintCoordinationPolicy::RequireIdentical;
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

    /** Coordinated prefix boundary and terminal-state contract. */
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

        /** @return Whether every required participant owns a common hit. */
        bool hit() const
        {
            return supported && cache_enabled && common_matched_tokens > 0;
        }
    };

    /** Reduction surface used to coordinate prefix metadata across ranks. */
    class IPrefixCollectiveCoordinator
    {
    public:
        virtual ~IPrefixCollectiveCoordinator() = default;

        /** Reduce one signed scalar with MIN. */
        virtual bool allMinInt(int local_value, int *global_value) = 0;
        /** Reduce one unsigned scalar with MIN. */
        virtual bool allMinUInt64(uint64_t local_value, uint64_t *global_value) = 0;
        /** Reduce one unsigned scalar with MAX. */
        virtual bool allMaxUInt64(uint64_t local_value, uint64_t *global_value) = 0;
        /** Reduce one boolean with logical AND. */
        virtual bool allAndBool(bool local_value, bool *global_value) = 0;
        /** Reduce one boolean with logical OR. */
        virtual bool allOrBool(bool local_value, bool *global_value) = 0;
    };

    /** MPI implementation of the prefix metadata reduction surface. */
    class MPIPrefixCollectiveCoordinator : public IPrefixCollectiveCoordinator
    {
    public:
        /** Bind coordination to one already-owned communicator. */
        explicit MPIPrefixCollectiveCoordinator(MPI_Comm communicator);

        /** @copydoc IPrefixCollectiveCoordinator::allMinInt */
        bool allMinInt(int local_value, int *global_value) override;
        /** @copydoc IPrefixCollectiveCoordinator::allMinUInt64 */
        bool allMinUInt64(uint64_t local_value, uint64_t *global_value) override;
        /** @copydoc IPrefixCollectiveCoordinator::allMaxUInt64 */
        bool allMaxUInt64(uint64_t local_value, uint64_t *global_value) override;
        /** @copydoc IPrefixCollectiveCoordinator::allAndBool */
        bool allAndBool(bool local_value, bool *global_value) override;
        /** @copydoc IPrefixCollectiveCoordinator::allOrBool */
        bool allOrBool(bool local_value, bool *global_value) override;

    private:
        MPI_Comm communicator_ = MPI_COMM_NULL;
    };

    /** Build one coordinator record from a participant-local cache lookup. */
    PrefixParticipantLookup makePrefixParticipantLookup(
        int participant_id,
        DeviceId device,
        const PrefixLookupResult &hit,
        std::string domain_id = {},
        uint64_t placement_epoch = 0,
        PrefixFingerprintCoordinationPolicy fingerprint_policy =
            PrefixFingerprintCoordinationPolicy::RequireIdentical);

    /** Clamp local and optional cross-rank reports to one safe prefix. */
    PrefixCoordinationResult coordinatePrefixLookups(
        std::vector<PrefixParticipantLookup> participants,
        IPrefixCollectiveCoordinator *domain_coordinator = nullptr);

    /** Project coordinated metadata back onto the cache lookup API. */
    PrefixLookupResult makePrefixLookupResult(
        const PrefixCoordinationResult &coordination,
        int block_size);

} // namespace llaminar2
