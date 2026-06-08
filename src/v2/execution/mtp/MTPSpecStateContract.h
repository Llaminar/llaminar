#pragma once

#include "MTPSpecDecodeMetadata.h"

#include <cstdint>
#include <string>
#include <vector>

namespace llaminar2
{

    struct MTPSpecStepPlan
    {
        int request_index = -1;
        int request_id = -1;

        int draft_count = 0;
        int target_rows = 0;
        int valid_sampled_count = 0;
        int committed_output_count = 0;
        int accepted_count = 0;
        int rejected_count = 0;

        int base_cached_tokens = 0;
        int target_cached_tokens = 0;
        int accepted_state_slot_index = kMTPSpecDecodeInvalidToken;

        int correction_replay_start_index = kMTPSpecDecodeInvalidToken;
        int correction_replay_count = 0;

        int bonus_ready_token_row = kMTPSpecDecodeInvalidToken;
        int bonus_ready_token_index = kMTPSpecDecodeInvalidToken;
        int bonus_ready_state_slot_index = kMTPSpecDecodeInvalidToken;

        int32_t next_condition_token = kMTPSpecDecodeInvalidToken;

        bool all_drafts_accepted = false;
        bool stopped = false;

        bool publishesAcceptedState() const { return accepted_count > 0; }
        bool requiresCorrectionReplay() const { return correction_replay_count > 0; }
        bool hasBonusReadyToken() const
        {
            return bonus_ready_token_row != kMTPSpecDecodeInvalidToken;
        }
        bool hasRejectedSuffix() const { return rejected_count > 0; }
    };

    struct MTPSpecStepPlanBatch
    {
        bool ok = false;
        std::string error;

        MTPSpecDecodeMetadataShape shape;
        int request_count = 0;
        std::vector<MTPSpecStepPlan> steps;
    };

    class IMTPSpecStateBackend
    {
    public:
        virtual ~IMTPSpecStateBackend() = default;

        virtual bool prepareSpecSlots(const MTPSpecStepPlan &plan) = 0;
        virtual bool runDraftGraph(const MTPSpecStepPlan &plan) = 0;
        virtual bool runTargetVerifierGraph(const MTPSpecStepPlan &plan) = 0;
        virtual bool publishAcceptedState(const MTPSpecStepPlan &plan) = 0;
        virtual bool discardRejectedState(const MTPSpecStepPlan &plan) = 0;
    };

    MTPSpecStepPlanBatch buildMTPSpecStepPlans(
        const MTPSpecDecodeMetadataBatch &batch,
        const MTPSpecDecodeStatePublicationPlan &publication_plan);

    MTPSpecStepPlanBatch buildMTPSpecStepPlans(
        const MTPSpecDecodeMetadataBatch &batch,
        const std::vector<int32_t> &base_cached_tokens);

} // namespace llaminar2
