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
        /**
         * @brief Whether this participant owns shifted MTP sidecar KV rows.
         *
         * SingleDevice and TP participants publish both main verifier state and
         * shifted MTP KV.  In LocalPP, non-final stages own main KV/GDN state
         * but do not own the sidecar cache; the final stage owns sidecar KV and
         * leaves this true.  This keeps publication explicit instead of making
         * non-final PP stages truncate sidecar state they never produced.
         */
        bool publish_mtp_shifted_kv = true;
        /**
         * @brief Whether the first accepted shifted KV row is already resident.
         *
         * Dense vLLM-style sidecars can leave the row corresponding to verifier
         * target row zero in the shifted MTP KV cache. MoE/non-preserving
         * sidecars restore that speculative row away, then explicitly publish
         * the same row from the verifier-base terminal hidden before this plan
         * reaches the KV publisher. In both cases, true means publication may
         * use the ordinary shifted target `target_cached_tokens - (depth + 1)`.
         */
        bool reuse_initial_mtp_shifted_kv_row = true;

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

    /**
     * @brief Common-prefix decision for one speculative decode step.
     *
     * Multi-device MTP must make one publication decision for the whole
     * topology.  A TP shard, PP stage, or ExpertParallel participant is not
     * allowed to publish a verifier state row that another participant cannot
     * publish.  This result records the minimum accepted prefix, a per-
     * participant clamped publication plan, and whether the clamping changed
     * any participant enough that the caller should replay from the common
     * prefix instead of publishing directly.
     */
    struct MTPSpecCommonStepPlan
    {
        bool ok = false;
        std::string error;

        int common_accepted_count = 0;
        bool all_participants_direct = false;
        bool requires_common_fallback_replay = false;

        std::vector<MTPSpecStepPlan> clamped_steps;
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

    /**
     * @brief Clamp participant-local MTP publication plans to one common prefix.
     *
     * The input plans must describe the same logical request and sampled token
     * stream.  The returned `clamped_steps` never publish more verifier state
     * than the smallest participant-local `accepted_count`.  If any participant
     * had to be shortened, `requires_common_fallback_replay` is set so the
     * topology coordinator can clear speculative local state and replay from
     * that common point.
     */
    MTPSpecCommonStepPlan coordinateMTPSpecCommonAcceptedPrefix(
        const std::vector<MTPSpecStepPlan> &participant_steps);

} // namespace llaminar2
