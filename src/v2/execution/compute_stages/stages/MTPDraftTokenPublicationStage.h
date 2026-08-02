/**
 * @file MTPDraftTokenPublicationStage.h
 * @brief Captured publication of one greedy MTP proposal into resident scratch.
 *
 * A speculative transaction repeatedly runs the MTP sidecar and publishes the
 * winning proposal token into a persistent draft slot.  This stage owns that
 * complete producer-to-consumer boundary: optional serial-equivalent branch
 * penalties are applied to the fresh sidecar row, then the deterministic
 * lowest-token-tie argmax writes both value and token to arena-owned device
 * storage.
 *
 * The stage is deliberately suitable for cloning into a device-controlled
 * generation loop.  It performs no allocation, transfer, event operation,
 * callback, host observation, or synchronization.  Its caller supplies the
 * exact non-null graph stream and publishes completion with an event outside
 * the captured node.
 */

#pragma once

#include "../IComputeStage.h"
#include "../StageParamsBase.h"

#include <cstddef>
#include <cstdint>
#include <string>

namespace llaminar2
{
    class IBackend;

    /**
     * @brief Score and publish one device-resident MTP draft proposal.
     *
     * The source logit row is freshly produced by the immediately preceding
     * sidecar graph.  When penalties are enabled, the row observes durable
     * generated-token counts, the transaction's condition token when it is not
     * already durable, and exactly `prior_draft_count` earlier proposal slots.
     * The subsequent argmax therefore matches the corresponding serial sampler
     * while keeping every intermediate byte on device.
     */
    class MTPDraftTokenPublicationStage final : public IComputeStage
    {
    public:
        /**
         * @brief Persistent bindings and immutable launch geometry.
         *
         * Every pointer names model- or arena-lifetime device memory.  The
         * destination slot and branch depth are graph identity because they are
         * embedded in captured kernel arguments.  Device contents remain replay
         * data and may change between launches without recapture.
         */
        struct Params
        {
            STAGE_PARAMS_COMMON_FIELDS;

            IBackend *backend = nullptr;
            float *logits_row_device = nullptr;
            int logits_row = 0;
            int vocab_size = 0;

            bool apply_penalties = false;
            const int32_t *first_condition_token_device = nullptr;
            const int32_t *prior_draft_tokens_device = nullptr;
            int prior_draft_count = 0;
            const int32_t *generated_token_counts_device = nullptr;
            void *penalty_policy_device = nullptr;
            float presence_penalty = 0.0F;
            float frequency_penalty = 0.0F;
            bool first_token_already_in_history = false;

            float *draft_values_device = nullptr;
            int32_t *draft_tokens_device = nullptr;
            int destination_slot = 0;
            float *argmax_partial_values_device = nullptr;
            int32_t *argmax_partial_indices_device = nullptr;
            int argmax_partial_capacity = 0;

            std::string stage_name = "mtp_draft_token_publication";
        };

        static_assert(StageParamsRequired<Params>);

        explicit MTPDraftTokenPublicationStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override
        {
            return ComputeStageType::MTP_DRAFT_TOKEN_PUBLICATION;
        }
        std::string name() const override { return params_.stage_name; }
        size_t estimatedFlops() const override;
        size_t estimatedMemoryBytes() const override;
        bool supportsBackend(ComputeBackendType backend) const override;
        bool isGraphCapturable() const override { return true; }
        bool isCollectiveStage() const override { return false; }
        CoherencePolicy coherencePolicy() const override
        {
            return CoherencePolicy::NONE;
        }
        StageDumpInfo buildDumpInfoImpl() const override;
        StageBufferContract bufferContract() const override;

        /**
         * @brief Compare every scalar and address embedded in captured nodes.
         *
         * Replay data is intentionally excluded.  A mismatch means the caller
         * must materialize another graph instead of silently reusing a graph
         * whose kernels target a different branch depth or destination slot.
         */
        [[nodiscard]] bool hasSameCaptureIdentity(
            const Params &other) const noexcept;

        [[nodiscard]] const Params &getParams() const noexcept
        {
            return params_;
        }

    private:
        [[nodiscard]] bool validate() const;

        Params params_;
    };
} // namespace llaminar2
