/**
 * @file MTPSpeculativeStatePublicationStage.h
 * @brief Captured device transaction that publishes one accepted MTP outcome.
 *
 * A grouped MTP verifier produces compact outcome metadata, captured recurrent
 * rows, routed-expert history, and speculative KV state.  Serial decode may
 * continue only after all of those objects have advanced to the same accepted
 * prefix.  This stage expresses that complete mutation as one graph-capturable
 * transaction over persistent device addresses.
 *
 * The stage deliberately owns no host mirror and performs no allocation,
 * transfer, synchronization, event creation, or callback during execute().
 * Host code may validate immutable geometry and establish an event edge before
 * launching the captured graph, but every per-transaction state mutation is a
 * kernel node in this stage's graph body.  That restriction makes the captured
 * graph safe to clone into a future device-controlled generation loop.
 */

#pragma once

#include "../IComputeStage.h"
#include "../StageParamsBase.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace llaminar2
{
    class IBackend;
    class IKVCache;
    class IMoEGroupedVerifierHistogramPublisher;
    namespace sampling_math
    {
        struct MTPCommittedVerifierIdentityRecord;
    }

    /**
     * @brief Publish compact verifier acceptance into every mutable model state.
     *
     * One instance represents one immutable capture geometry.  Dynamic values,
     * including accepted counts, restore rows, cache lengths, condition tokens,
     * and generation-controller budgets, are read from persistent device rows
     * on every replay.  Pointer identity and scalar launch geometry therefore
     * form the capture identity; changing either requires a new stage/graph.
     */
    class MTPSpeculativeStatePublicationStage final : public IComputeStage
    {
    public:
        /**
         * @brief Persistent checkpoint used to restore one request's main KV metadata.
         *
         * The checkpoint is populated on the verifier stream before verifier
         * execution.  Publication derives every layer's accepted ring head and
         * count from this immutable pre-verifier base, so a partially restored
         * cache cannot become visible to the next decode transaction.
         */
        struct MainKVBinding
        {
            IKVCache *cache = nullptr;
            int first_sequence_index = 0;
            const void *base_checkpoint_device = nullptr;
            size_t base_checkpoint_bytes = 0;

            [[nodiscard]] bool valid() const noexcept
            {
                return cache != nullptr && first_sequence_index >= 0 &&
                       base_checkpoint_device != nullptr &&
                       base_checkpoint_bytes > 0;
            }

            bool operator==(const MainKVBinding &) const = default;
        };

        /**
         * @brief Immutable bindings and launch geometry captured by the stage.
         *
         * All pointer members name model- or arena-lifetime device storage.  No
         * pointer may refer to a host vector, request stack object, or temporary
         * allocation.  The orchestrator validates and freezes this structure
         * before graph warmup/capture.
         */
        struct Params
        {
            STAGE_PARAMS_COMMON_FIELDS;

            IBackend *backend = nullptr;

            const int32_t *outcome_tokens_device = nullptr;
            int *outcome_meta_device = nullptr;
            int outcome_token_stride = 0;
            int outcome_meta_stride = 0;

            const int32_t *base_cached_tokens_device = nullptr;
            int32_t *accepted_restore_rows_device = nullptr;
            int32_t *target_cached_tokens_device = nullptr;
            int32_t *accepted_state_counts_device = nullptr;
            int32_t *publication_ok_flags_device = nullptr;
            int32_t *next_condition_tokens_device = nullptr;
            int32_t *all_drafts_accepted_flags_device = nullptr;
            int32_t *stopped_flags_device = nullptr;

            /** Canonical first-token bank consumed by the next verifier. */
            int32_t *next_verifier_condition_tokens_device = nullptr;

            /** Next-transaction inputs consumed directly by the full sidecar. */
            int32_t *next_sidecar_condition_tokens_device = nullptr;
            int32_t *next_sidecar_position_ids_device = nullptr;

            bool generation_controller_owned = false;
            int32_t *generation_response_tokens_device = nullptr;
            int generation_response_token_stride = 0;
            int *generation_control_device = nullptr;
            int generation_control_stride = 0;

            /** Reusable verifier row consumed by the transaction being committed. */
            const int32_t *verifier_input_tokens_device = nullptr;
            /** Physical request-row capacity of @ref verifier_input_tokens_device. */
            int verifier_input_token_stride = 0;
            /** Durable identity written atomically with controller/response commit. */
            sampling_math::MTPCommittedVerifierIdentityRecord *
                committed_verifier_identity_device = nullptr;

            int request_count = 0;
            int verifier_rows_per_request = 0;
            int max_state_commit_rows = 0;

            std::vector<IMoEGroupedVerifierHistogramPublisher *>
                moe_histogram_publishers;
            /**
             * Table-owned stream shared by every deferred MoE publisher.
             *
             * Null is valid only when @ref moe_histogram_publishers is empty.
             * The graph cache borrows this immutable identity and therefore
             * cannot create a late producer after histogram topology sealing.
             */
            void *moe_histogram_publication_stream = nullptr;
            std::vector<MainKVBinding> main_kv_bindings;

            bool publish_shifted_kv = false;
            std::vector<IKVCache *> shifted_kv_caches;
            int32_t *shifted_target_cached_tokens_device = nullptr;
            int32_t *shifted_accepted_state_counts_device = nullptr;

            void *penalty_policy_device = nullptr;
            int32_t *generated_token_counts_device = nullptr;
            int vocab_size = 0;

            std::vector<IComputeStage *> verifier_state_stages;
            bool require_captured_verifier_state = false;

            std::string stage_name =
                "mtp_speculative_state_publication";
        };

        static_assert(StageParamsRequired<Params>);

        explicit MTPSpeculativeStatePublicationStage(Params params);

        /**
         * @brief Admit every deferred MoE histogram writer before capture.
         *
         * The accepted-state graph borrows the exact table-owned stream used by
         * all retained per-layer route ledgers. Each publisher certifies that
         * immutable identity here; execute() then performs only capture-safe
         * validation and kernel enqueue operations.
         */
        bool prepareGraphLaunch(IDeviceContext *ctx, void *stream) override;

        /**
         * @brief Require exact-stream producer admission before native capture.
         *
         * The accepted-state graph retains one immutable capture stream for its
         * lifetime.  Histogram maintenance allocates the producer rendezvous
         * events once while that graph is materialized; replay needs no host
         * preparation because the stream identity cannot change without a new
         * graph family.
         */
        GraphLaunchPreparationPolicy graphLaunchPreparationPolicy() const override
        {
            return GraphLaunchPreparationPolicy::CaptureOnly;
        }

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override
        {
            return ComputeStageType::MTP_SPEC_STATE_PUBLICATION;
        }
        std::string name() const override { return params_.stage_name; }
        size_t estimatedFlops() const override { return 0; }
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
         * @brief Compare every value embedded in the captured launch sequence.
         *
         * Device contents are intentionally excluded: they are replay inputs.
         * Pointers, vector membership/order, and geometry are included because
         * CUDA/HIP graph nodes retain them by value. Request penalty values and
         * the evolving pending-condition predicate live only in the pointed-to
         * device policy, so changing either never creates another graph family.
         */
        [[nodiscard]] bool hasSameCaptureIdentity(
            const Params &other) const noexcept;

        const Params &getParams() const noexcept { return params_; }

    private:
        [[nodiscard]] bool validate() const;
        [[nodiscard]] bool publishMoEHistograms(void *stream) const;
        [[nodiscard]] bool publishMainKV(void *stream) const;
        [[nodiscard]] bool publishShiftedKV(void *stream) const;
        [[nodiscard]] bool publishPenaltyHistory(void *stream) const;
        [[nodiscard]] bool publishVerifierState(void *stream) const;

        Params params_;
    };

} // namespace llaminar2
