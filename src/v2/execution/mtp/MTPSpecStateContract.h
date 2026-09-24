/**
 * @file MTPSpecStateContract.h
 * @brief Typed planning contract for speculative verifier-state publication.
 *
 * The metadata decoder describes what a grouped verifier observed; this file
 * turns that observation into an immutable, backend-independent publication
 * plan.  CPU execution also carries the exact host-owned verifier token row in
 * the plan, while GPU execution keeps that identity in resident controller
 * storage.  Publishers consume these plans without reconstructing lifecycle
 * state from counters, cache lengths, or backend-specific flags.
 */
#pragma once

#include "MTPSpecDecodeMetadata.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace llaminar2
{

    /**
     * @brief Exact verifier row whose successful transaction is being committed.
     *
     * Host-owned CPU execution carries this immutable identity through the same
     * typed publication plan as KV and recurrent state. Device-owned GPU
     * execution may omit it because its captured controller publishes the
     * identity directly from resident verifier input storage. An absent value
     * therefore means "not host-authoritative", never an inferred token row.
     */
    struct MTPSpecVerifierInputIdentity
    {
        int draft_depth = 0;
        std::vector<int32_t> verifier_input_tokens;

        /** @return Whether the identity is a complete `[target, drafts...]` row. */
        [[nodiscard]] bool valid() const
        {
            if (draft_depth <= 0 ||
                verifier_input_tokens.size() !=
                    static_cast<size_t>(draft_depth + 1))
            {
                return false;
            }
            for (const int32_t token : verifier_input_tokens)
            {
                if (token < 0)
                    return false;
            }
            return true;
        }
    };

    /**
     * @brief Complete publication decision for one request in one MTP epoch.
     *
     * Every field is derived before state mutation begins.  It names the
     * accepted verifier prefix, correction/bonus suffix, target cache lengths,
     * and the authority responsible for shifted-MTP KV publication.
     */
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

        /** Exact CPU-owned verifier transaction evidence, when host-authoritative. */
        std::optional<MTPSpecVerifierInputIdentity> verifier_input_identity;

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

        /** @return Whether at least one verifier state row becomes live. */
        [[nodiscard]] bool publishesAcceptedState() const
        {
            return accepted_count > 0;
        }

        /** @return Whether the rejected suffix requires a correction row. */
        [[nodiscard]] bool requiresCorrectionReplay() const
        {
            return correction_replay_count > 0;
        }

        /** @return Whether an all-accepted transaction retained a bonus token. */
        [[nodiscard]] bool hasBonusReadyToken() const
        {
            return bonus_ready_token_row != kMTPSpecDecodeInvalidToken;
        }

        /** @return Whether any proposed draft row was rejected. */
        [[nodiscard]] bool hasRejectedSuffix() const
        {
            return rejected_count > 0;
        }
    };

    /** @brief Validated per-request plans produced from one metadata batch. */
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
     * topology. A TP shard, PP stage, or routed-expert participant is not
     * allowed to publish a verifier state row that another participant cannot
     * publish. This result records the minimum accepted prefix, a per-
     * participant clamped publication plan, and whether any participant-local
     * suffix was shortened. A shortened suffix is not a replay request: callers
     * publish the clamped common prefix and discard the unshared speculative
     * metadata so the whole topology advances to one serial-decode boundary.
     */
    struct MTPSpecCommonStepPlan
    {
        bool ok = false;
        std::string error;

        int common_accepted_count = 0;
        bool all_participants_direct = false;
        bool clamped_participant_suffix = false;

        std::vector<MTPSpecStepPlan> clamped_steps;
    };

    /**
     * @brief Mutation surface used by the generic speculative-state driver.
     *
     * Implementations own either host state (CPU) or device state (GPU).  The
     * driver calls these methods in transaction order; an implementation must
     * return false before publishing a partial state transition it cannot
     * complete.
     */
    class IMTPSpecStateBackend
    {
    public:
        /** @brief Allow polymorphic backend destruction. */
        virtual ~IMTPSpecStateBackend() = default;

        /** @brief Reserve or reset speculative slots named by @p plan. */
        virtual bool prepareSpecSlots(const MTPSpecStepPlan &plan) = 0;

        /** @brief Execute the predictor graph for @p plan. */
        virtual bool runDraftGraph(const MTPSpecStepPlan &plan) = 0;

        /** @brief Execute the grouped target verifier for @p plan. */
        virtual bool runTargetVerifierGraph(const MTPSpecStepPlan &plan) = 0;

        /** @brief Atomically make the accepted verifier prefix live. */
        virtual bool publishAcceptedState(const MTPSpecStepPlan &plan) = 0;

        /** @brief Retire every speculative row excluded by @p plan. */
        virtual bool discardRejectedState(const MTPSpecStepPlan &plan) = 0;
    };

    /**
     * @brief Build publication plans from decoded metadata and explicit targets.
     * @param batch Validated verifier metadata for every request.
     * @param publication_plan Canonical main/shifted cache publication targets.
     * @return Complete plans, or one diagnostic explaining why none are usable.
     */
    MTPSpecStepPlanBatch buildMTPSpecStepPlans(
        const MTPSpecDecodeMetadataBatch &batch,
        const MTPSpecDecodeStatePublicationPlan &publication_plan);

    /**
     * @brief Build plans using base cache lengths and derived target lengths.
     * @param batch Validated verifier metadata for every request.
     * @param base_cached_tokens Main-model cache length before each verifier.
     * @return Complete plans, or one diagnostic explaining why none are usable.
     */
    MTPSpecStepPlanBatch buildMTPSpecStepPlans(
        const MTPSpecDecodeMetadataBatch &batch,
        const std::vector<int32_t> &base_cached_tokens);

    /**
     * @brief Clamp participant-local MTP publication plans to one common prefix.
     *
     * The input plans must describe the same logical request and sampled token
     * stream.  The returned `clamped_steps` never publish more verifier state
     * than the smallest participant-local `accepted_count`. If any participant
     * had to be shortened, `clamped_participant_suffix` is set and the returned
     * steps have all unshared bonus/correction suffix metadata cleared. This
     * makes the clamped plans directly publishable as the grouped production path.
     */
    MTPSpecCommonStepPlan coordinateMTPSpecCommonAcceptedPrefix(
        const std::vector<MTPSpecStepPlan> &participant_steps);

} // namespace llaminar2
