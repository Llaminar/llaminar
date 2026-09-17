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
         * @brief Immutable authority of this captured publication transaction.
         *
         * CompactOutcome derives metadata from an already reduced outcome.
         * BoundedGeneration additionally commits the request's response budget.
         * PipelineFollower consumes the tail's committed metadata and publishes
         * only its own main-model state. It cannot own response, predictor or
         * penalty-history storage. Unbound is rejected before any mutation.
         */
        enum class Authority
        {
            Unbound,
            CompactOutcome,
            BoundedGeneration,
            PipelineFollower,
        };

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

            /** @return Whether the owner supplied a complete persistent checkpoint binding. */
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

            /** Frozen authority belongs in capture identity, never request data. */
            Authority authority = Authority::Unbound;
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

        /** @brief Freeze ownership and addresses; execute rejects incomplete bindings.
         * @param params Exact persistent bindings and publication authority. */
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
         * @brief Fence asynchronous histogram maintenance around native capture.
         *
         * Every deferred publisher lends this stage the same table-owned
         * stream. Entering increments each table's capture activity before the
         * backend begins recording; Completed or Aborted releases those
         * references only after the backend has closed capture. A partially
         * accepted Entering edge is unwound in reverse order.
         *
         * @param ctx Device context owning @p stream.
         * @param stream Exact immutable publication/capture stream.
         * @param transition Typed native-capture lifecycle edge.
         * @return true when every publisher accepted the transition.
         */
        bool transitionGraphCaptureActivity(
            IDeviceContext *ctx,
            void *stream,
            GraphCaptureActivityTransition transition) override;

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

        /** @brief Enqueue exactly this authority's publication DAG on the bound stream.
         * @param ctx Participant-local execution context.
         * @return False before mutation for invalid ownership, or on enqueue failure. */
        bool execute(IDeviceContext *ctx) override;
        /** @return Canonical stage classification for captured accepted-state publication. */
        ComputeStageType type() const override
        {
            return ComputeStageType::MTP_SPEC_STATE_PUBLICATION;
        }
        /** @return Caller-qualified diagnostic identity. */
        std::string name() const override { return params_.stage_name; }
        /** @return Publication moves state; it performs no floating-point arithmetic. */
        size_t estimatedFlops() const override { return 0; }
        /** @return Diagnostic metadata traffic estimate, not an allocation ledger. */
        size_t estimatedMemoryBytes() const override;
        /** @return Whether the backend provides this device-resident publication protocol. */
        bool supportsBackend(ComputeBackendType backend) const override;
        /** @return All runtime mutations are fixed native graph nodes. */
        bool isGraphCapturable() const override { return true; }
        /** @return Transport is an explicit surrounding graph edge, never hidden here. */
        bool isCollectiveStage() const override { return false; }
        /** @return Raw persistent bindings need no tensor coherence transition in this stage. */
        CoherencePolicy coherencePolicy() const override
        {
            return CoherencePolicy::NONE;
        }
        /** @return Immutable geometry and authority for diagnostics. */
        StageDumpInfo buildDumpInfoImpl() const override;
        /** @return Only the arena slots this authority actually reads or writes. */
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

        /** @return Immutable bindings; pointed-to device values remain runtime-owned. */
        const Params &getParams() const noexcept { return params_; }

        /** @brief Validate immutable ownership before constructing collective edges.
         * @return Whether authority, local owners and capture geometry are complete.
         * This reads no device values and enqueues nothing. Graph builders use
         * the same admission as execute so malformed follower bindings cannot
         * enter a collective before the local publication rejects them. */
        [[nodiscard]] bool validate() const;

    private:
        /** @brief Derive/commit only for outcome owners; followers consume existing metadata.
         * @param stream Exact non-null capture or execution stream.
         * @return Whether the authority-specific metadata operation was enqueued. */
        [[nodiscard]] bool publishAcceptanceMetadata(void *stream) const;
        /** @brief Publish local accepted router rows on their pre-admitted producer stream. */
        [[nodiscard]] bool publishMoEHistograms(void *stream) const;
        /** @brief Restore each local KV owner from its own pre-verifier checkpoint. */
        [[nodiscard]] bool publishMainKV(void *stream) const;
        /** @brief Publish predictor KV owned by the outcome authority, never a follower. */
        [[nodiscard]] bool publishShiftedKV(void *stream) const;
        /** @brief Commit the outcome owner's penalty history once from accepted output. */
        [[nodiscard]] bool publishPenaltyHistory(void *stream) const;
        /** @brief Select local recurrent-state rows using the committed device indices. */
        [[nodiscard]] bool publishVerifierState(void *stream) const;

        Params params_;
    };

} // namespace llaminar2
