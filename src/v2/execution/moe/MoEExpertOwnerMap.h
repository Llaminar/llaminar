/**
 * @file MoEExpertOwnerMap.h
 * @brief Graph-native routed MoE whole-expert owner map.
 */

#pragma once

#include "MoEExpertParallelPlan.h"
#include "backends/DeviceId.h"
#include "backends/GlobalDeviceAddress.h"

#include <cstddef>
#include <string>
#include <vector>

namespace llaminar2
{
    struct MoEExpertOwner
    {
        int layer_idx = -1;
        int expert_id = -1;
        int tier_idx = -1;
        int owner_participant = -1;
        DeviceId device = DeviceId::cpu();
        bool resident = false;

        std::string tier_name;
        std::string domain_name;
        int domain_participant_index = -1;
        int owner_world_rank = -1;
        bool owner_world_rank_known = false;
        GlobalDeviceAddress address;
    };

    struct MoEExpertOwnerParticipant
    {
        int participant_id = -1;
        int tier_idx = -1;
        std::string tier_name;
        std::string domain_name;
        int domain_participant_index = -1;
        GlobalDeviceAddress address;
        DeviceId device = DeviceId::invalid();
        int world_rank = -1;
        bool world_rank_known = false;
    };

    struct MoEExpertOwnerMapBuildOptions
    {
        bool reject_tensor_parallel_experts = true;
    };

    class MoEExpertOwnerMap
    {
    public:
        static MoEExpertOwnerMap build(
            const MoEExpertParallelPlan &plan,
            const MoEExpertOwnerMapBuildOptions &options = {});

        const std::vector<MoEExpertOwner> &owners() const { return owners_; }
        const std::vector<MoEExpertOwnerParticipant> &participants() const { return participants_; }

        const MoEExpertOwner *ownerFor(int layer_idx, int expert_id) const;
        const MoEExpertOwnerParticipant *participantForId(int participant_id) const;

        std::vector<int> participantIdsForTier(int tier_idx) const;
        std::vector<int> expertsForParticipant(int layer_idx, int owner_participant) const;
        std::vector<bool> expertMaskForParticipant(
            int layer_idx,
            int owner_participant,
            int num_experts) const;
        size_t ownerCountForExpert(int layer_idx, int expert_id) const;

    private:
        std::vector<MoEExpertOwner> owners_;
        std::vector<MoEExpertOwnerParticipant> participants_;
    };

} // namespace llaminar2