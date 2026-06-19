/**
 * @file IInferenceRunner.h
 * @brief Interface for inference execution
 * @author David Sanftenberg
 * @date December 2025
 *
 * Interface implemented by DeviceGraphOrchestrator for inference execution.
 */

#pragma once

#include <array>
#include <optional>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <stdexcept>

#include "../../../backends/DeviceId.h"
#include "../../mtp/MTPRejectionSampler.h"
#include "../../prefix_cache/PrefixCacheStateProbe.h"
#include "../../prefix_cache/PrefixStateSnapshot.h"

namespace llaminar2
{
    // Forward declarations
    class TensorBase;
    struct PlacementPlan;
    struct GraphExecutorStats;
    struct SamplingParams;
    struct LogitPenalty;
    struct MTPSpecStepPlan;
    struct MTPSpecStepPlanBatch;
    struct MTPSpecDecodeVerifierInputPlan;
    struct PrefillChunkSchedulerPolicy;
    class MoERebalanceController;

    /**
     * @brief Lightweight view of a device runner's local logits state
     *
     * Returned by getLogitsLocalInfo() to provide GPU pointer, device, and
     * shape information for column-parallel LM head gather and GPU-side sampling.
     * This decouples RankOrchestrator from DeviceGraphOrchestrator's
     * internal InferenceState struct.
     */
    struct LogitsLocalInfo
    {
        const void *gpu_ptr = nullptr;  ///< GPU buffer pointer (nullptr if CPU-only)
        std::optional<DeviceId> device; ///< GPU device for backend lookup
        size_t vocab_local = 0;         ///< Local vocab size (columns in logits_local)
        TensorBase *tensor = nullptr;   ///< Tensor pointer for CPU fallback (data())
        void *stream = nullptr;         ///< Explicit GPU stream (must match forward pass stream)

        // Device-resident scratch for the multi-block argmax reduction (owned by
        // the runner's BufferArena). Supplied to IBackend::argmaxF32() so the
        // CUDA two-pass reduction never has to allocate on the hot path. Null /
        // zero capacity means GPU-side argmax is unavailable for this runner.
        void *argmax_partial_vals = nullptr; ///< FP32 scratch [argmax_partial_capacity]
        void *argmax_partial_idxs = nullptr; ///< INT32 scratch [argmax_partial_capacity]
        int argmax_partial_capacity = 0;     ///< Number of entries in the scratch buffers

        /// True if this info is valid (has a tensor)
        explicit operator bool() const { return tensor != nullptr; }
    };

    enum class DeviceLogitsSource : uint8_t
    {
        Main,
        MTP,
        AllPosition
    };

    enum class DeviceDistributionBuffer : uint8_t
    {
        Target,
        Draft
    };

    struct DeviceSpeculativeVerifyResult
    {
        int32_t token = -1;
        bool accepted = false;
        float accept_probability = 0.0f;
        float accept_threshold = 0.0f;
    };

    using DeviceSpeculativeVerifyBatchOutcome =
        MTPDeviceRejectionBatchOutcome;

    /**
     * @brief Device-resident compact outcome buffers for stochastic MTP.
     *
     * This is the first-class handoff object for the vLLM-style path where the
     * verifier summary remains on GPU.  The pointed-to buffers are owned by the
     * runner and are valid until the runner stages another stochastic outcome
     * request.  Future publication code can consume these buffers directly and
     * avoid the per-step D2H boundary; current compatibility callers can pass
     * the handle to materializeDeviceSpeculativeOutcomesForHostResponse().
     */
    struct DeviceSpeculativeOutcomeHandle
    {
        const int32_t *output_tokens_device = nullptr;
        const int *meta_device = nullptr;
        int request_count = 0;
        int output_token_stride = sampling_math::kSpeculativeBatchMaxOutputTokens;
        int meta_stride = sampling_math::kSpeculativeBatchMetaCount;
        DeviceId device;
        void *stream = nullptr;
        /**
         * @brief Event recorded after compact verifier outcome rows are ready.
         *
         * State publication may enqueue more work on @ref stream after this
         * handle is returned.  Host response materialization should wait on
         * this event from its own explicit bridge stream so a response copy
         * does not accidentally wait for later live-state publication work.
         */
        std::shared_ptr<void> response_ready_event;
        /**
         * @brief Optional timing event recorded before compact outcome reduction.
         *
         * These events exist only when structured perfstats are enabled.  They
         * measure the GPU reducer/summary kernels themselves and deliberately
         * exclude upstream verifier graph replay that is already queued before
         * the reducer starts.
         */
        std::shared_ptr<void> producer_start_timing_event;
        /// Optional timing event recorded immediately after compact reduction.
        std::shared_ptr<void> producer_stop_timing_event;

        bool valid() const
        {
            return output_tokens_device != nullptr &&
                   meta_device != nullptr &&
                   request_count > 0 &&
                   output_token_stride >= sampling_math::kSpeculativeBatchMaxOutputTokens &&
                   meta_stride >= sampling_math::kSpeculativeBatchMetaCount &&
                   stream != nullptr &&
                   response_ready_event != nullptr;
        }
    };

    /**
     * @brief Host-known shape data for publishing from a device outcome row.
     *
     * The compact verifier outcome tells the backend how many speculative rows
     * were accepted and which tokens should be exposed.  It does not carry the
     * structural invariants that are already known to the runner: request count
     * and padded verifier width.  The pre-verifier cache length is normally
     * snapshotted into device metadata before verifier replay; callers should
     * pass @ref base_cached_tokens_device only when they already own an
     * equivalent compact device array.
     */
    struct DeviceSpeculativePublicationRequest
    {
        DeviceSpeculativeOutcomeHandle outcome;
        int request_count = 0;
        int max_draft_tokens = 0;
        /**
         * @brief Optional compact device array of pre-verifier cache lengths.
         *
         * Most single-device callers let the runner snapshot its live KV count
         * into the persistent metadata workspace immediately before verifier
         * graph replay.  Future batched or TP callers may pass an already-owned
         * device array here.  Implementations must not upload
         * @ref base_cached_tokens during resident publication; the host vector is
         * retained only for legacy diagnostics and should be empty on the GPU
         * hot path.
         */
        const int32_t *base_cached_tokens_device = nullptr;
        std::vector<int32_t> base_cached_tokens;
        int base_sidecar_position = 0;
        bool publish_mtp_shifted_kv = true;

        bool valid() const
        {
            return outcome.valid() &&
                   request_count > 0 &&
                   outcome.request_count == request_count &&
                   max_draft_tokens > 0 &&
                   base_sidecar_position >= 0;
        }
    };

    /**
     * @brief Device-resident logical sequence state produced by MTP publication.
     *
     * The old host-facing getters (`get_position()` and `sequence_lengths()`)
     * return scalar/vector snapshots and therefore cannot represent a
     * graph-captured publication without a D2H sync.  This handle is the
     * resident counterpart: it names the device buffers that hold the next
     * logical position, sequence length, next condition token, and validity flag
     * for each request in the active speculative batch.
     *
     * The pointed-to buffers are owned by the runner's workspace.  They are
     * valid only until the runner resets request state or stages a newer
     * speculative publication mailbox.  Consumers must enqueue work on `stream`
     * or explicitly wait on it; nullptr/default streams are not valid.  The
     * accepted-state count is the device-owned replay boundary: it is the first
     * output-token index that still needs correction-token replay after a
     * stochastic rejection.
     */
    struct DeviceResidentLogicalSequenceStateHandle
    {
        const int32_t *target_positions_device = nullptr;
        const int32_t *target_sequence_lengths_device = nullptr;
        const int32_t *accepted_state_counts_device = nullptr;
        const int32_t *next_condition_tokens_device = nullptr;
        const int32_t *all_drafts_accepted_flags_device = nullptr;
        const int32_t *stopped_flags_device = nullptr;
        const int32_t *publication_ok_flags_device = nullptr;
        int request_count = 0;
        DeviceId device = DeviceId::invalid();
        void *stream = nullptr;
        void *ready_event = nullptr;
        uint64_t live_state_epoch = 0;

        bool valid() const
        {
            return target_positions_device != nullptr &&
                   target_sequence_lengths_device != nullptr &&
                   accepted_state_counts_device != nullptr &&
                   next_condition_tokens_device != nullptr &&
                   all_drafts_accepted_flags_device != nullptr &&
                   stopped_flags_device != nullptr &&
                   publication_ok_flags_device != nullptr &&
                   request_count > 0 &&
                   device.is_valid() &&
                   stream != nullptr &&
                   ready_event != nullptr;
        }

        /**
         * @brief Return whether this handle contains a row for @p request_index.
         *
         * Phase 10 consumers should call this before deriving row pointers.  It
         * keeps bounds checks paired with handle validity, which is especially
         * important while logical positions are still migrating from host-owned
         * scalars to device-resident metadata.
         */
        bool coversRequest(int request_index) const
        {
            return valid() &&
                   request_index >= 0 &&
                   request_index < request_count;
        }

        /**
         * @brief Return whether two handles name the same resident mailbox.
         *
         * Phase 10 prelaunch and continuation paths may carry a handle across
         * one served-output boundary.  Matching every stream/event/pointer
         * field prevents a later request, workspace rebind, or reset from
         * accidentally reusing an old sidecar replay.
         */
        bool sameMailboxAs(
            const DeviceResidentLogicalSequenceStateHandle &other) const
        {
            return valid() &&
                   other.valid() &&
                   target_positions_device == other.target_positions_device &&
                   target_sequence_lengths_device ==
                       other.target_sequence_lengths_device &&
                   accepted_state_counts_device ==
                       other.accepted_state_counts_device &&
                   next_condition_tokens_device ==
                       other.next_condition_tokens_device &&
                   all_drafts_accepted_flags_device ==
                       other.all_drafts_accepted_flags_device &&
                   stopped_flags_device == other.stopped_flags_device &&
                   publication_ok_flags_device ==
                       other.publication_ok_flags_device &&
                   request_count == other.request_count &&
                   device == other.device &&
                   stream == other.stream &&
                   ready_event == other.ready_event &&
                   live_state_epoch == other.live_state_epoch;
        }

        /// Device row containing the next logical position for one request.
        const int32_t *targetPositionDeviceForRequest(int request_index) const
        {
            return coversRequest(request_index)
                       ? target_positions_device + request_index
                       : nullptr;
        }

        /// Device row containing the next sequence length for one request.
        const int32_t *targetSequenceLengthDeviceForRequest(int request_index) const
        {
            return coversRequest(request_index)
                       ? target_sequence_lengths_device + request_index
                       : nullptr;
        }

        /// Device row containing the accepted-state count for one request.
        const int32_t *acceptedStateCountDeviceForRequest(int request_index) const
        {
            return coversRequest(request_index)
                       ? accepted_state_counts_device + request_index
                       : nullptr;
        }

        /// Device row containing the next condition token for one request.
        const int32_t *nextConditionTokenDeviceForRequest(int request_index) const
        {
            return coversRequest(request_index)
                       ? next_condition_tokens_device + request_index
                       : nullptr;
        }

        /// Device row containing whether the verifier accepted every draft.
        const int32_t *allDraftsAcceptedFlagDeviceForRequest(int request_index) const
        {
            return coversRequest(request_index)
                       ? all_drafts_accepted_flags_device + request_index
                       : nullptr;
        }

        /// Device row containing whether the emitted output hit a stop token.
        const int32_t *stoppedFlagDeviceForRequest(int request_index) const
        {
            return coversRequest(request_index)
                       ? stopped_flags_device + request_index
                       : nullptr;
        }

        /// Device row containing the publication validity flag for one request.
        const int32_t *publicationOkFlagDeviceForRequest(int request_index) const
        {
            return coversRequest(request_index)
                       ? publication_ok_flags_device + request_index
                       : nullptr;
        }
    };

    /**
     * @brief Host-mirror adoption request for a resident speculative publish.
     *
     * Device-resident publication makes accelerator metadata authoritative for
     * the next live logical position and sequence length.  The host still owns
     * legacy mirrors used by graph signatures and API diagnostics, so it needs
     * one explicit adoption handoff.  This request carries the resident mailbox
     * plus the pre-verifier base cache lengths that remain host-known from
     * scheduling; it does not carry compact verifier outcome rows.
     */
    struct DeviceResidentHostStateAdoptionRequest
    {
        DeviceResidentLogicalSequenceStateHandle logical_state;
        std::vector<int32_t> base_cached_tokens;
        bool publish_mtp_shifted_kv = true;

        bool valid() const
        {
            return logical_state.valid() &&
                   logical_state.request_count > 0 &&
                   static_cast<int>(base_cached_tokens.size()) ==
                       logical_state.request_count;
        }
    };

    /**
     * @brief One logical request inside a device-side stochastic MTP batch.
     *
     * The descriptor is intentionally value-owned: thresholds and stop tokens
     * are copied into fixed-size arrays before the runner call. That keeps the
     * request-batch handoff atomic and avoids dangling pointers when the caller
     * builds several requests before a GPU reducer consumes them.  Seeded
     * vLLM-style verification can set @ref derive_thresholds_from_seed to let
     * the GPU derive accept/residual thresholds from
     * @ref inverse_sample_seed and @ref inverse_sample_first_logical_position
     * instead of capturing host scalar thresholds in the verifier launch.
     */
    struct DeviceStochasticBatchOutcomeRequest
    {
        int request_id = -1;          ///< Logical request id for diagnostics.
        int first_target_slot = -1;   ///< First verifier target row slot.
        int first_draft_slot = -1;    ///< First sampled draft-token slot.
        int row_count = 0;            ///< Number of speculative rows to compare.
        int32_t first_token = -1;     ///< First main-model token, if host-owned.
        bool first_token_from_device = false; ///< Read first token from sample slot.
        int first_target_sample_slot = -1;    ///< Device first-token slot.
        int bonus_target_slot = -1;           ///< Bonus row slot, or -1.
        float bonus_threshold = 0.0f;         ///< RNG draw for bonus sampling.
        uint64_t inverse_sample_seed = 0;     ///< vLLM rejection inverse RNG seed.
        int inverse_sample_first_logical_position = 0;
        bool use_vllm_probability_rejection = false;
        bool derive_thresholds_from_seed = false;
        bool use_device_draft_tokens = true; ///< Null host draft pointer when true.
        std::array<int32_t, sampling_math::kSpeculativeBatchMaxRows> draft_tokens;
        std::array<float, sampling_math::kSpeculativeBatchMaxRows> accept_thresholds;
        std::array<float, sampling_math::kSpeculativeBatchMaxRows> residual_thresholds;
        std::array<int32_t, sampling_math::kSpeculativeBatchMaxStopTokens> stop_tokens;
        int stop_token_count = 0;

        DeviceStochasticBatchOutcomeRequest()
        {
            draft_tokens.fill(-1);
            accept_thresholds.fill(0.0f);
            residual_thresholds.fill(0.0f);
            stop_tokens.fill(-1);
        }

        const int32_t *hostDraftTokensOrNull() const
        {
            return use_device_draft_tokens ? nullptr : draft_tokens.data();
        }
    };

    /**
     * @brief Lightweight view of a captured snapshot with 2D shape metadata
     *
     * Returned by getSnapshotWithShape() to provide shape information
     * alongside the raw FP32 data pointer. The shape (rows, cols) comes
     * from the stage's getDumpInfo() at capture time, so stages own their
     * own dimension reporting.
     */
    struct SnapshotInfo
    {
        const float *data = nullptr; ///< Pointer to FP32 snapshot data (not owned)
        size_t size = 0;             ///< Total element count (rows * cols)
        size_t rows = 0;             ///< Logical rows (e.g. seq_len)
        size_t cols = 0;             ///< Logical cols (e.g. hidden_dim, kv_dim, d_ff)

        explicit operator bool() const { return data != nullptr && size > 0; }
    };

    /**
     * @brief Optional high-level decode-step result for orchestration-aware callers.
     *
     * Low-level runners expose forward() plus explicit sampling. Runners wrapped
     * around IOrchestrationRunner can expose decodeStep() so benchmark mode
     * measures MTP, rollback, and decode-boundary maintenance through the same
     * path as normal generation.
     */
    struct DecodeStepOutput
    {
        std::vector<int32_t> tokens;
        bool is_complete = false;
        std::string error;
    };

    /**
     * @brief Optional high-level batched decode-step result for benchmark lanes.
     *
     * Each entry in `tokens_by_request` contains the accepted tokens for the
     * matching logical request in the active request batch. The benchmark treats
     * `n_predict` as a per-request target and reports aggregate emitted tokens
     * across the batch, so request-batched lanes measure amortized verifier cost
     * without pretending they are single-request decode.
     */
    struct DecodeBatchStepOutput
    {
        std::vector<std::vector<int32_t>> tokens_by_request;
        std::vector<bool> is_complete_by_request;
        std::string error;
    };

    /**
     * @brief Execution path type
     */
    enum class ExecutionPath
    {
        PIPELINE, ///< Traditional imperative pipeline (Qwen2Pipeline)
        GRAPH     ///< Graph-based execution (DeviceGraphOrchestrator)
    };

    /**
     * @brief Interface for inference execution
     *
     * Implemented by DeviceGraphOrchestrator for transformer model inference.
     */
    class IInferenceRunner
    {
    public:
        virtual ~IInferenceRunner() = default;

        // =====================================================================
        // Core Inference API
        // =====================================================================

        /**
         * @brief Run forward pass
         *
         * @param tokens Token IDs
         * @param seq_len Sequence length
         * @return true if forward succeeded
         */
        virtual bool forward(const int *tokens, int seq_len) = 0;

        /**
         * @brief Run a single-batch forward pass from device-resident token IDs.
         *
         * @param token_shadow Host copy of the same token IDs for bookkeeping,
         *        logging, and cache metadata. GPU embedding execution must read
         *        from `token_ids_device`, not from this host pointer.
         * @param token_ids_device Stable device pointer to INT32 token IDs.
         *        The pointer must remain valid for any cached graph replay that
         *        the runner enables for this shape.
         * @param seq_len Sequence length for this single-batch forward.
         * @return true when the forward pass succeeds.
         */
        virtual bool forwardWithDeviceTokenIds(
            const int *token_shadow,
            const void *token_ids_device,
            int seq_len)
        {
            (void)token_shadow;
            (void)token_ids_device;
            (void)seq_len;
            return false;
        }

        /**
         * @brief Run a padded batched forward pass from device-resident token IDs.
         *
         * `token_batches` is the logical host shadow for diagnostics and state
         * bookkeeping. `token_ids_device` is the execution source of truth and
         * must point at a caller-owned flat INT32 buffer laid out as
         * `[request_count, padded_seq_len]`. Padding tokens in that device
         * buffer must match the host shadow materialization performed by the
         * caller so graph row indices and sequence-length masks stay aligned.
         *
         * The default hard-fails unsupported runners. Rank-level orchestrators
         * need a per-participant pointer bundle rather than one raw pointer, so
         * this contract is intentionally runner-local until that ownership is
         * modeled explicitly.
         *
         * @param token_batches Logical per-request host shadow tokens.
         * @param token_ids_device Stable flat device pointer for padded tokens.
         * @param padded_seq_len Row width of `token_ids_device`.
         * @return true when the batched forward pass succeeds.
         */
        virtual bool forwardBatchWithDeviceTokenIds(
            const std::vector<std::vector<int>> &token_batches,
            const void *token_ids_device,
            int padded_seq_len)
        {
            (void)token_batches;
            (void)token_ids_device;
            (void)padded_seq_len;
            return false;
        }

        /**
         * @brief Prepare the compact all-position verifier input token row on device.
         *
         * vLLM-style stochastic verification already has sampled draft tokens in a
         * runner-owned device buffer.  GPU runners can use this hook to build the
         * verifier input sequence `[accepted_main_token, draft_0, ...]` in another
         * arena-owned device buffer and then pass that pointer to
         * forwardWithDeviceTokenIds().  The host `token_shadow` still exists for
         * metadata and diagnostics, but the embedding graph reads the device row.
         *
         * @param first_token The already-sampled main-model token at verifier row 0.
         * @param first_draft_slot First slot in the runner-owned sampled-draft buffer.
         * @param draft_token_count Number of draft tokens to copy after `first_token`.
         * @param total_verifier_input_tokens Total verifier forward sequence length.
         * @return Stable device pointer on success, nullptr if unsupported or invalid.
         */
        virtual const void *prepareMTPVerifierInputTokensOnDevice(
            int32_t first_token,
            int first_draft_slot,
            int draft_token_count,
            int total_verifier_input_tokens)
        {
            (void)first_token;
            (void)first_draft_slot;
            (void)draft_token_count;
            (void)total_verifier_input_tokens;
            return nullptr;
        }

        /**
         * @brief Prepare a complete verifier input row on device from host-known tokens.
         *
         * This hook is intentionally for deterministic parity and diagnostic
         * checks that already own the whole verifier token row as fixture data.
         * Production MTP verification should prefer
         * prepareMTPVerifierInputTokensOnDevice() or
         * prepareMTPVerifierInputTokensOnDeviceFromDeviceFirstToken() so the hot
         * path stays device-owned after sampling. Implementations must still
         * materialize this row on the verifier graph stream, not on the device
         * default stream, because the embedding graph consumes the returned
         * pointer during captured replay.
         *
         * @param verifier_tokens Host-known compact verifier row.
         * @param total_verifier_input_tokens Number of valid entries in the row.
         * @param draft_token_count Number of draft tokens represented by the row.
         * @return Stable device pointer on success, nullptr if unsupported or invalid.
         */
        virtual const void *prepareMTPVerifierInputTokensOnDeviceFromHostRow(
            const int32_t *verifier_tokens,
            int total_verifier_input_tokens,
            int draft_token_count)
        {
            (void)verifier_tokens;
            (void)total_verifier_input_tokens;
            (void)draft_token_count;
            return nullptr;
        }

        /**
         * @brief Whether this runner can execute a prepared bucketed prefill
         *        chunk schedule for the current request state.
         */
        virtual bool supportsPrefillChunkSchedule(int seq_len) const
        {
            (void)seq_len;
            return false;
        }

        /**
         * @brief Execute a prompt/suffix through prepared bucketed prefill chunks.
         *
         * Default false keeps the path opt-in for runners that can preserve KV,
         * logits, and maintenance state across chunk boundaries.
         */
        virtual bool forwardPrefillChunkSchedule(
            const int *tokens,
            int seq_len,
            const PrefillChunkSchedulerPolicy &policy,
            int pad_token_id,
            bool allow_padded_execution)
        {
            (void)tokens;
            (void)seq_len;
            (void)policy;
            (void)pad_token_id;
            (void)allow_padded_execution;
            return false;
        }

        /**
         * @brief Get logits from last forward pass
         *
         * @return Pointer to logits [vocab_size], or nullptr if unavailable
         */
        virtual const float *logits() const = 0;

        virtual bool forwardMTP(int32_t draft_condition_token)
        {
            (void)draft_condition_token;
            return false;
        }

        /**
         * @brief Run one MTP sidecar row for a following device-side sampler.
         *
         * Default runners fall back to the normal synchronized sidecar path.
         * GPU graph runners can override this to request deferred final sync
         * and expose the sidecar stream through the device distribution path.
         */
        virtual bool forwardMTPForDeviceSampling(int32_t draft_condition_token)
        {
            return forwardMTP(draft_condition_token);
        }

        /**
         * @brief True when the runner can chain depth-0 MTP sidecar calls.
         *
         * Chained drafts use the hidden state produced by the previous sidecar
         * as the next sidecar's terminal-hidden input, while appending shifted
         * MTP KV rows at explicit logical positions. Runners that cannot keep
         * this state contract should return false so callers hard-fail for
         * draft depths greater than one.
         */
        virtual bool supportsChainedMTPDrafts() const { return false; }

        /**
         * @brief True when forwardMTPAndSampleGreedy() can provide a runner-native
         *        combined sidecar/sample path.
         */
        virtual bool supportsMTPSidecarSampleFusion() const { return false; }

        /**
         * @brief True when sidecar logits can flow into device sampling without an immediate host sync.
         *
         * A supporting runner may execute forwardMTPForDeviceSampling() and leave
         * MTP logits ordered on an explicit pending stream. The next
         * buildStochasticDistributionOnDevice(DeviceLogitsSource::MTP, ...)
         * consumes that stream so distribution construction, compact sampling,
         * and the final scalar D2H copy form one ordered chain.
         *
         * This is a partial vLLM-style step: it removes a sidecar completion sync,
         * but the sampled token may still return to the host until the sidecar
         * embedding input accepts a device-resident token source.
         */
        virtual bool supportsMTPSidecarLogitsStreamHandoff() const { return false; }

        /**
         * @brief True when a chained MTP sidecar can consume sampled draft tokens on device.
         *
         * This is the next step after logits-stream handoff: the sampler writes
         * draft token IDs into arena/workspace memory, and the following sidecar
         * embedding reads that device slot directly. A runner that returns true
         * must not upload the same token through a host pointer for the sidecar
         * embedding path.
         */
        virtual bool supportsMTPDeviceDraftTokenInput() const { return false; }

        /**
         * @brief True when MTP sidecar execution is isolated from main live state.
         *
         * Runners that return true guarantee forwardMTP*() may append speculative
         * MTP-side shifted KV rows and mutate MTP scratch/logits, but it does not
         * advance or otherwise mutate the main verifier state: main KV cache,
         * main hybrid/GDN state, main logits, terminal hidden, positions, or
         * decode bookkeeping. The caller may then skip restoring the verifier
         * base checkpoint after a sidecar draft and rely on explicit shifted-row
         * commit/truncate calls to discard speculative MTP rows.
         */
        virtual bool supportsMTPSidecarPreservesMainState() const { return false; }

        /**
         * @brief True when a token-count-only verifier-base checkpoint is safe.
         *
         * KV-only decoder models can sometimes synthesize the MTP verifier base
         * as logical token counts because restoring those counts is enough to
         * recreate decode-equivalent state. Hybrid/recurrent models such as GDN
         * also need payload state snapshots, so they must return false even if
         * their sidecar execution itself preserves main state.
         */
        virtual bool supportsLogicalMTPVerifierBaseCheckpoint() const { return false; }

        /**
         * @brief True when the first shifted MTP KV row produced by the sidecar
         *        can be reused as the accepted-row commit.
         *
         * This is intentionally stricter than supportsMTPSidecarPreservesMainState().
         * A runner may keep the main verifier state isolated while still requiring
         * the verifier/terminal-hidden publication path to append the accepted
         * shifted MTP KV row. MoE runners use that stricter path because routed
         * expert state publication is authoritative only after the target verifier
         * has selected the accepted row.
         */
        virtual bool supportsMTPShiftedRowReuseFromSidecar() const { return false; }

        /**
         * @brief True when the verifier path must replay accepted tokens through
         *        normal one-token decode to preserve mutable model state exactly.
         *
         * Stateful architectures such as hybrid GDN may produce all-position
         * verifier logits whose token rows look plausible while the final
         * recurrent/conv state is not byte-for-byte decode-equivalent unless
         * verifier-row state is explicitly published. Returning true means the
         * runner needs the common sequential verifier replay path when it does
         * not also advertise supportsMTPSpecStatePublication().
         */
        virtual bool requiresMTPDecodeEquivalentVerifierReplay() const { return false; }

        /**
         * @brief Exact verifier-row proof surface advertised by this runner.
         *
         * The row-count fields are deliberately separate from the older boolean
         * shortcuts.  A caller that wants to publish state from a batched
         * all-position verifier must check a direct-all-position lane for the
         * model family and row count it is about to use.  A caller that only
         * needs the shared one-token replay oracle checks the decode-equivalent
         * lane instead.
         */
        virtual MTPVerifierRowCapability mtpVerifierRowCapability() const
        {
            return {};
        }

        /**
         * @brief Phase 9.8 economical verifier capability advertised by runner.
         *
         * This is intentionally separate from mtpVerifierRowCapability(). A
         * runner may be numerically decode-equivalent only because it replays
         * rows serially; that is correct, but not the grouped, resident,
         * graph-capturable fast path required before Phase 10 rollout claims.
         */
        virtual MTPVerifierEconomyCapability mtpVerifierEconomyCapability() const
        {
            return {};
        }

        /**
         * @brief True when the runner implements vLLM-style accepted-count
         *        publication from the most recent target verifier graph.
         */
        virtual bool supportsMTPSpecStatePublication() const { return false; }

        /**
         * @brief True when accepted-state publication can consume a compact
         *        stochastic verifier outcome without copying it to host first.
         *
         * This is intentionally independent of supportsMTPSpecStatePublication().
         * The older method answers whether the runner may choose the direct
         * all-position verifier policy.  This method answers whether an already
         * proven grouped verifier outcome can be published without a host bridge:
         * compact outcome metadata drives accepted-row selection, KV truncation,
         * and terminal hidden/state restoration on an explicit device stream.
         */
        virtual bool supportsDeviceResidentMTPSpecStatePublication() const { return false; }

        /**
         * @brief Publish the accepted verifier state prefix into live model state.
         *
         * Implementations must fail loudly unless the most recent verifier graph
         * is the graph that produced `plan.draft_count` all-position state rows.
         * `plan.target_rows` is the metadata transaction length, including the
         * bonus-ready sampled-token slot. GPU implementations must use an
         * explicit non-null stream for every publication kernel.
         */
        virtual bool publishAcceptedMTPSpecState(
            const MTPSpecStepPlan &plan,
            std::string *error = nullptr)
        {
            (void)plan;
            if (error)
                *error = "runner does not support MTP spec-state publication";
            return false;
        }

        /**
         * @brief Publish accepted verifier state for a request batch.
         *
         * Request-batched verifier graphs use one padded graph execution for
         * several logical requests. Implementations must map each request's
         * accepted row to the physical verifier graph row, publish KV/state for
         * the matching request index, and fail loudly if the last forward graph
         * is not the graph that produced the batch. The default preserves the
         * single-request contract and refuses real batches.
         */
        virtual bool publishAcceptedMTPSpecStateBatch(
            const MTPSpecStepPlanBatch &plans,
            std::string *error = nullptr)
        {
            (void)plans;
            if (error)
                *error = "runner does not support batched MTP spec-state publication";
            return false;
        }

        /**
         * @brief Publish accepted verifier state from a device-resident outcome.
         *
         * This is Phase 10's no-D2H state-publication boundary.  Implementations
         * must consume @p request.outcome.meta_device and
         * @p request.outcome.output_tokens_device on @p request.outcome.stream,
         * then leave live model state exactly as publishAcceptedMTPSpecStateBatch()
         * would for the equivalent host step plans.  Host copies may still occur
         * later to flush response tokens, but they must not be required for live
         * state mutation once this method succeeds.
         */
        virtual bool publishAcceptedMTPSpecStateBatchFromDeviceOutcome(
            const DeviceSpeculativePublicationRequest &request,
            std::string *error = nullptr)
        {
            (void)request;
            if (error)
            {
                *error =
                    "runner does not support device-resident MTP spec-state publication";
            }
            return false;
        }

        /**
         * @brief Refresh host-visible logical positions after resident publication.
         *
         * Device-resident MTP publication mutates live KV/recurrent state from
         * compact GPU metadata before the compatibility host bridge flushes
         * output tokens.  Once the bridge has produced the equivalent
         * MTPSpecStepPlanBatch, callers can use this method to make
         * get_position() and sequence_lengths() reflect the already-published
         * device state without invoking publishAcceptedMTPSpecStateBatch() a
         * second time.
         *
         * Implementations must update host mirrors only. They must not append,
         * truncate, restore, synchronize a GPU stream, or mutate cache/state
         * that was already published from device metadata.
         */
        virtual bool adoptDeviceResidentMTPSpecPublishedHostState(
            const MTPSpecStepPlanBatch &plans,
            std::string *error = nullptr)
        {
            (void)plans;
            if (error)
            {
                *error =
                    "runner does not support adopting device-resident MTP host state";
            }
            return false;
        }

        /**
         * @brief Refresh host mirrors directly from resident device metadata.
         *
         * This is the request-batched GPU hot-path counterpart to
         * adoptDeviceResidentMTPSpecPublishedHostState().  Implementations
         * should wait on @p request.logical_state.ready_event using an explicit
         * bridge stream, copy only the small logical-state arrays needed for
         * host mirrors, and update host-side positions/cache heads without
         * reconstructing an MTPSpecStepPlanBatch from compact verifier outcomes.
         */
        virtual bool adoptDeviceResidentMTPSpecPublishedHostStateFromDeviceMetadata(
            const DeviceResidentHostStateAdoptionRequest &request,
            std::string *error = nullptr)
        {
            (void)request;
            if (error)
            {
                *error =
                    "runner does not support adopting device-resident MTP host state from device metadata";
            }
            return false;
        }

        /**
         * @brief Run a chained MTP sidecar step from the previous sidecar hidden.
         *
         * @param draft_condition_token Token whose shifted MTP KV row is appended.
         * @param position_id Logical shifted-cache position for the append.
         */
        virtual bool forwardMTPFromLastDraft(int32_t draft_condition_token, int position_id)
        {
            (void)draft_condition_token;
            (void)position_id;
            return false;
        }

        /**
         * @brief Chained sidecar variant of forwardMTPForDeviceSampling().
         */
        virtual bool forwardMTPFromLastDraftForDeviceSampling(
            int32_t draft_condition_token,
            int position_id)
        {
            return forwardMTPFromLastDraft(draft_condition_token, position_id);
        }

        /**
         * @brief Run a chained MTP sidecar from a sampled draft token slot.
         *
         * @param draft_sample_slot Slot in the runner-owned device draft-token
         *        buffer written by sampleStochasticDistributionOnDevice(Draft, ...).
         * @param position_id Logical shifted-cache position for the append.
         * @return true when the graph ran and left logits ready for sampling.
         *
         * The default hard-fails by returning false; callers should gate this
         * with supportsMTPDeviceDraftTokenInput() and never silently fall back to
         * a host token upload in the vLLM-style path.
         */
        virtual bool forwardMTPFromDeviceDraftForDeviceSampling(
            int draft_sample_slot,
            int position_id)
        {
            (void)draft_sample_slot;
            (void)position_id;
            return false;
        }

        /**
         * @brief Run the first MTP sidecar from a sampled main-model token slot.
         *
         * Penalty-free stochastic GPU decoding can sample the first token into
         * runner-owned device memory and defer the host read until the batched
         * verifier summary. This entry point feeds that token directly to the
         * sidecar embedding. The default hard-fails; GPU implementations must
         * provide explicit stream ordering before advertising the path.
         */
        virtual bool forwardMTPFromDeviceTargetForDeviceSampling(
            int target_sample_slot,
            int position_id)
        {
            (void)target_sample_slot;
            (void)position_id;
            return false;
        }

        /**
         * @brief Run the first MTP sidecar from a target slot and sample a draft slot.
         *
         * This is the fused greedy companion to
         * forwardMTPFromDeviceTargetForDeviceSampling().  The first main-model
         * token is already stored in a runner-owned target-token slot, so the
         * sidecar embedding must consume that device value directly.  The MTP
         * draft proposal is then sampled into @p draft_sample_slot for the
         * verifier input row.  The optional host shadow in @p out_token is for
         * response planning only; the verifier source of truth remains the
         * device draft slot.
         */
        virtual bool forwardMTPFromDeviceTargetAndSampleGreedyToDeviceDraftSlot(
            int target_sample_slot,
            int position_id,
            int draft_sample_slot,
            int32_t *out_token)
        {
            if (!out_token)
                return false;
            if (!forwardMTPFromDeviceTargetForDeviceSampling(
                    target_sample_slot,
                    position_id))
            {
                return false;
            }
            return sampleGreedyFromMTPLogitsToDeviceDraftSlot(
                draft_sample_slot,
                out_token);
        }

        /**
         * @brief Run the first MTP sidecar from a resident logical-state mailbox.
         *
         * This is the Phase 10 bridge between device-side accepted-state
         * publication and the next sidecar replay.  The condition token and
         * logical position row come from @p logical_state, so callers do not
         * need to read `get_position()` or sampled tokens back to the CPU before
         * starting the next speculative step.  Implementations must verify that
         * the handle belongs to their runner, wait on its readiness event using
         * an explicit stream, and fail hard if any part of the handoff is stale.
         */
        virtual bool forwardMTPFromDeviceResidentLogicalStateForDeviceSampling(
            const DeviceResidentLogicalSequenceStateHandle &logical_state,
            int request_index = 0)
        {
            (void)logical_state;
            (void)request_index;
            return false;
        }

        /**
         * @brief Run a sidecar step and greedily sample its logits as one logical
         *        operation.
         *
         * GPU runners may use this to keep captured sidecar replay and the argmax
         * reduction on one stream, avoiding an intermediate host synchronization.
         * The default implementation preserves the existing synchronous contract.
         */
        virtual bool forwardMTPAndSampleGreedy(int32_t draft_condition_token, int32_t *out_token)
        {
            if (!out_token)
                return false;
            if (!forwardMTP(draft_condition_token))
                return false;
            const int token = sampleGreedyFromMTPLogitsOnDevice();
            if (token < 0)
                return false;
            *out_token = token;
            return true;
        }

        /**
         * @brief Run a first-depth sidecar and greedily sample into a device slot.
         *
         * @p out_token is the host-visible planning shadow.  The sampled token
         * must also be written to @p draft_sample_slot in the runner-owned
         * device draft-token arena so later verifier input construction can
         * consume it without re-uploading the host shadow.
         */
        virtual bool forwardMTPAndSampleGreedyToDeviceDraftSlot(
            int32_t draft_condition_token,
            int draft_sample_slot,
            int32_t *out_token)
        {
            if (!out_token)
                return false;
            if (!forwardMTP(draft_condition_token))
                return false;
            return sampleGreedyFromMTPLogitsToDeviceDraftSlot(
                draft_sample_slot,
                out_token);
        }

        /**
         * @brief Run one first-depth MTP sidecar row for several request slots.
         *
         * `draft_condition_tokens[i]` is the already-sampled main-model token
         * for request `i`, and `position_ids[i]` is that request's live decode
         * position before the sidecar append. Implementations must execute a
         * true request batch (`batch_size=request_batch`, `seq_len=1`) and write
         * one greedy draft token per request to `out_tokens`.
         *
         * The default hard-fails so benchmark/server paths cannot accidentally
         * claim request batching while looping scalar sidecars.
         */
        virtual bool forwardMTPBatchAndSampleGreedy(
            const int32_t *draft_condition_tokens,
            const int *position_ids,
            int request_batch,
            int32_t *out_tokens)
        {
            (void)draft_condition_tokens;
            (void)position_ids;
            (void)request_batch;
            (void)out_tokens;
            return false;
        }

        /**
         * @brief Run first-depth batched MTP and write sampled drafts to device slots.
         *
         * This is the request-batched, vLLM-style companion to
         * forwardMTPBatchAndSampleGreedy().  Implementations enqueue one true
         * batched sidecar graph (`batch_size=request_batch, seq_len=1`) and
         * write the greedy proposal token for request `i` into
         * `STOCHASTIC_DRAFT_SAMPLE_TOKENS[first_draft_slot + i * slot_stride]`
         * on an explicit stream.  Depth-one callers pass `slot_stride=1`;
         * deeper request batches pass the draft depth so slots stay
         * request-major for the verifier.  `out_tokens` is an optional host
         * shadow for metadata paths
         * that have not yet been promoted to fully device-resident publication;
         * verifier execution must consume the device slots, not re-upload this
         * shadow.
         *
         * The default hard-fails so callers cannot silently replace a batched
         * device-token path with host token staging.
         */
        virtual bool forwardMTPBatchAndSampleGreedyToDeviceDraftSlots(
            const int32_t *draft_condition_tokens,
            const int *position_ids,
            int request_batch,
            int first_draft_slot,
            int slot_stride,
            int32_t *out_tokens)
        {
            (void)draft_condition_tokens;
            (void)position_ids;
            (void)request_batch;
            (void)first_draft_slot;
            (void)slot_stride;
            (void)out_tokens;
            return false;
        }

        /**
         * @brief Run one chained MTP sidecar row for several request slots.
         *
         * This is the request-batched counterpart of
         * `forwardMTPFromLastDraftAndSampleGreedy()`.  It consumes the previous
         * batched MTP hidden rows, appends one shifted-cache row per request at
         * `position_ids[i]`, and returns one greedy draft token per request.
         * The default hard-fails so deeper request-batched drafting cannot
         * silently devolve into scalar loops.
         */
        virtual bool forwardMTPBatchFromLastDraftAndSampleGreedy(
            const int32_t *draft_condition_tokens,
            const int *position_ids,
            int request_batch,
            int32_t *out_tokens)
        {
            (void)draft_condition_tokens;
            (void)position_ids;
            (void)request_batch;
            (void)out_tokens;
            return false;
        }

        /**
         * @brief Run a chained batched MTP row and write proposals to device slots.
         *
         * This is the depth>1 companion to
         * forwardMTPBatchAndSampleGreedyToDeviceDraftSlots().  It consumes the
         * previous batched MTP hidden rows and writes request `i` to
         * `first_draft_slot + i * slot_stride`, preserving the request-major
         * verifier slot layout without a host-to-device staging pass.
         */
        virtual bool forwardMTPBatchFromLastDraftAndSampleGreedyToDeviceDraftSlots(
            const int32_t *draft_condition_tokens,
            const int *position_ids,
            int request_batch,
            int first_draft_slot,
            int slot_stride,
            int32_t *out_tokens)
        {
            (void)draft_condition_tokens;
            (void)position_ids;
            (void)request_batch;
            (void)first_draft_slot;
            (void)slot_stride;
            (void)out_tokens;
            return false;
        }

        /**
         * @brief Chained sidecar variant of forwardMTPAndSampleGreedy().
         */
        virtual bool forwardMTPFromLastDraftAndSampleGreedy(
            int32_t draft_condition_token,
            int position_id,
            int32_t *out_token)
        {
            if (!out_token)
                return false;
            if (!forwardMTPFromLastDraft(draft_condition_token, position_id))
                return false;
            const int token = sampleGreedyFromMTPLogitsOnDevice();
            if (token < 0)
                return false;
            *out_token = token;
            return true;
        }

        /**
         * @brief Chained sidecar plus greedy sample into a device draft slot.
         *
         * This preserves fixed-depth greedy response planning while making the
         * device slot the verifier source of truth.
         */
        virtual bool forwardMTPFromLastDraftAndSampleGreedyToDeviceDraftSlot(
            int32_t draft_condition_token,
            int position_id,
            int draft_sample_slot,
            int32_t *out_token)
        {
            if (!out_token)
                return false;
            if (!forwardMTPFromLastDraft(draft_condition_token, position_id))
                return false;
            return sampleGreedyFromMTPLogitsToDeviceDraftSlot(
                draft_sample_slot,
                out_token);
        }

        /**
         * @brief Flush deferred sidecar GPU work before checkpoint/verifier reads.
         *
         * Graph-captured MTP sidecars may defer their final stream sync so a
         * fused sampler can run on the same stream. Callers that need to read
         * or checkpoint sidecar-mutated KV/GDN state before sampling must make
         * that ordering explicit through this hook.
         */
        virtual bool flushPendingMTPWork() { return true; }

        /**
         * @brief Materialize the latest main-forward terminal hidden row for checkpointing.
         *
         * MTP restore/replay snapshots are only coherent when the terminal
         * hidden buffer belongs to the same logical state as KV/GDN/position
         * metadata. GPU runners keep that row in a stable sidecar input buffer
         * so graph-captured sidecars have a fixed buffer signature. Callers
         * must invoke this hook before capturing a live MTP checkpoint that may
         * later seed sidecar replay or commit verification.
         *
         * The default implementation is a no-op for runners without MTP live
         * state. Implementations that support MTP must fail hard if no current
         * main-forward hidden row can be materialized.
         */
        virtual bool ensureMTPCheckpointTerminalHidden() { return true; }

        /**
         * @brief Opt in to deferring the next all-position verifier graph sync.
         *
         * GPU runners may use this for greedy all-position MTP verification:
         * the verifier graph replays on its capture stream, then the greedy row
         * sampler is enqueued on that same stream and performs the required
         * synchronization through its device-to-host token copy. This is a
         * narrow vLLM-style stream handoff; stochastic verification keeps the
         * default synchronized boundary until its distribution path has the
         * same persistent stream contract.
         */
        virtual void setMTPAllPositionVerifierSyncDeferralEnabled(bool enabled)
        {
            (void)enabled;
        }

        /**
         * @brief Opt in to deferring the next MTP main condition-forward sync.
         *
         * This is a one-shot stream handoff for MTP: the main decode graph
         * replays on its capture stream, then the first-token GPU sampler or
         * stochastic distribution builder consumes logits on that same stream.
         * Runners that do not implement the handoff keep the synchronized
         * boundary by ignoring the request.
         */
        virtual void setMTPMainDecodeSyncDeferralEnabled(bool enabled)
        {
            (void)enabled;
        }

        /**
         * @brief Commit shifted MTP KV rows from the most recent main forward.
         *
         * MTP decode calls forwardMTP() before verifier/replay; that sidecar
         * step already appends the shifted row for tokens[0]. After the main
         * verifier or replay forward produces hidden rows for the accepted
         * token sequence, this method appends any remaining shifted rows so
         * the depth-0 MTP KV cache returns to the main_position - 1 invariant.
         *
         * @param tokens Accepted/correction tokens from the last main forward.
         * @param token_count Number of accepted/correction tokens.
         * @param already_appended_tokens Prefix of tokens already represented
         *        by the speculative sidecar KV append.
         */
        virtual bool commitMTPShiftedRowsFromLastForward(
            const int32_t *tokens,
            int token_count,
            int already_appended_tokens)
        {
            (void)tokens;
            return token_count <= already_appended_tokens;
        }

        /**
         * @brief Commit shifted MTP rows when the usable verifier hidden rows
         *        cover only a prefix of the emitted token sequence.
         *
         * Verifier-row restore can reuse mutable state for an accepted prefix
         * and then replay only a rejected correction suffix. In that case the
         * current hidden buffer may cover fewer rows than the logical token
         * span used to compute the shifted-cache position. Callers can pass an
         * explicit position_offset_override to keep those contracts separate.
         *
         * already_appended_tokens is a verifier-row indexing count. Most paths
         * have the same number of shifted KV rows resident already, but
         * non-reusable sidecar paths restore row zero away before verifier
         * publication. In that case pass already_appended_shifted_kv_tokens=0
         * while keeping already_appended_tokens=1 so hidden-row selection still
         * starts at verifier row zero.
         *
         * Set allow_speculative_discard only when the current MTP sidecar cache
         * is known to contain speculative rows produced by verifier-owned draft
         * steps. Generic callers should leave it false so unexpected extra rows
         * remain a hard failure.
         */
        virtual bool commitMTPShiftedRowsFromPartialForward(
            const int32_t *tokens,
            int token_count,
            int already_appended_tokens,
            int main_forward_token_count,
            bool allow_speculative_discard = false,
            int position_offset_override = -1,
            int already_appended_shifted_kv_tokens = -1)
        {
            (void)main_forward_token_count;
            (void)allow_speculative_discard;
            (void)position_offset_override;
            (void)already_appended_shifted_kv_tokens;
            return commitMTPShiftedRowsFromLastForward(
                tokens,
                token_count,
                already_appended_tokens);
        }

        /**
         * @brief Append one shifted MTP KV row using the current terminal hidden.
         *
         * This is used by decode-equivalent replay paths that advance the main
         * model one token at a time. The shifted sidecar row for tokens[i] must
         * be produced before tokens[i] is forwarded by the main model, while
         * the current terminal hidden still represents tokens[i - 1].
         */
        virtual bool commitMTPShiftedRowFromCurrentTerminalHidden(
            int32_t token,
            int already_appended_tokens,
            bool allow_speculative_discard = false,
            int position_offset_override = -1)
        {
            (void)token;
            (void)already_appended_tokens;
            (void)allow_speculative_discard;
            (void)position_offset_override;
            return false;
        }

        /**
         * @brief Append one shifted MTP KV row from a device-resident target token.
         *
         * Penalty-free stochastic GPU decode can defer the first main-token host
         * read.  The initial shifted-cache repair after verifier-base restore
         * must still append that token's row, so supporting runners read the
         * token from the same target sample slot used by
         * forwardMTPFromDeviceTargetForDeviceSampling().
         */
        virtual bool commitMTPShiftedRowFromDeviceTargetSample(
            int target_sample_slot,
            int already_appended_tokens,
            bool allow_speculative_discard = false,
            int position_offset_override = -1)
        {
            (void)target_sample_slot;
            (void)already_appended_tokens;
            (void)allow_speculative_discard;
            (void)position_offset_override;
            return false;
        }

        /**
         * @brief Append one shifted MTP KV row from a resident logical-state mailbox.
         *
         * Device-resident publication derives the next condition token on the
         * verifier stream.  Rejection repair can use that token directly instead
         * of copying the compact outcome to host first, but only while the
         * mailbox still belongs to the live runner state.  Implementations must
         * validate ownership, wait on the mailbox readiness event using an
         * explicit stream, and fail hard for stale handles.
         *
         * GPU implementations must also treat @p position_offset_override as
         * part of the resident-state handoff.  If it is negative, the method
         * must fail instead of deriving a position from get_position() or
         * sequence_lengths(), because those host mirrors can legitimately be
         * stale until adoptDeviceResidentMTPSpecPublishedHostState() runs.
         */
        virtual bool commitMTPShiftedRowFromDeviceResidentLogicalState(
            const DeviceResidentLogicalSequenceStateHandle &logical_state,
            int request_index,
            int already_appended_tokens,
            bool allow_speculative_discard = false,
            int position_offset_override = -1)
        {
            (void)logical_state;
            (void)request_index;
            (void)already_appended_tokens;
            (void)allow_speculative_discard;
            (void)position_offset_override;
            return false;
        }

        virtual const float *mtpLogits() const
        {
            return nullptr;
        }

        virtual bool setComputeAllPositionLogits(bool enabled)
        {
            (void)enabled;
            return false;
        }

        /**
         * @brief Enable compact row-indexed all-position verifier logits.
         *
         * This must preserve the all-position verifier state-publication
         * contract; it changes only logits production by packing selected
         * hidden rows before LM head. `row_count` is a graph-shape parameter.
         */
        virtual bool setComputeRowIndexedAllPositionLogits(bool enabled, int row_count)
        {
            (void)enabled;
            (void)row_count;
            return false;
        }

        /**
         * @brief Publish the current compact verifier row plan to the runner.
         *
         * The runner stores this host-side plan until the next row-indexed
         * all-position verifier forward. Device runners upload the row metadata
         * from this plan into their graph workspace immediately before execution
         * on the exact stream used by the cached graph.
         */
        virtual bool setMTPSpecVerifierInputPlan(
            const MTPSpecDecodeVerifierInputPlan &plan)
        {
            (void)plan;
            return true;
        }

        /**
         * @brief Clear any pending compact verifier row plan.
         *
         * Callers use this after the verifier forward, including failure paths,
         * so stale row metadata cannot leak into a later cached replay.
         */
        virtual void clearMTPSpecVerifierInputPlan() {}

        virtual const float *getAllPositionLogits() const
        {
            return nullptr;
        }

        /**
         * @brief Check if this runner has column-parallel local all-position logits.
         *
         * Used by MTP verification when TP participants each own a vocabulary shard
         * for every verified position.
         */
        virtual bool hasAllPositionLogitsLocal() const { return false; }

        /**
         * @brief Get local all-position verifier logits info for TP gathering.
         */
        virtual LogitsLocalInfo getAllPositionLogitsLocalInfo() const { return {}; }

        /**
         * @brief Get local all-position verifier logits info for a sampling consumer.
         *
         * Graph-captured verifier replay can hand off a producer stream that
         * owns the freshly-written all-position logits.  Rank-level TP sampling
         * must consume that stream once and reuse it for every sampled row in
         * the verifier batch; otherwise child shards can be read on an
         * unrelated stream before replay has completed.
         */
        virtual LogitsLocalInfo consumeAllPositionLogitsLocalInfoForSampling()
        {
            return getAllPositionLogitsLocalInfo();
        }

        virtual std::string mtpDecodeUnsupportedReason() const
        {
            return {};
        }

        /**
         * @brief True when this runner coordinates MTP draft-token choices across its domain.
         *
         * Multi-rank MTP decode requires every participant to verify and replay the
         * same draft sequence. Runners that span an MPI or TP domain should return
         * true only when their MTP sampling methods broadcast or otherwise agree on
         * MTP draft and verifier tokens.
         */
        virtual bool supportsMTPTokenCoordination() const { return false; }

        /**
         * @brief Sample the current MTP sidecar logits in greedy mode.
         *
         * Returns -1 when unavailable; callers may fall back to mtpLogits() on
         * single-rank paths. Multi-rank runners should coordinate the returned
         * token across all participants.
         */
        virtual int sampleGreedyFromMTPLogitsOnDevice() { return -1; }

        /**
         * @brief Sample MTP logits greedily and leave the token in a device slot.
         *
         * The returned host token is still needed by the response planner, but
         * the runner must also write @p draft_sample_slot in the same
         * device-resident draft-token arena consumed by
         * prepareMTPVerifierInputTokensOnDevice().  This prevents the compact
         * verifier from uploading a host shadow of a token that was just
         * produced on the GPU.
         */
        virtual bool sampleGreedyFromMTPLogitsToDeviceDraftSlot(
            int draft_sample_slot,
            int32_t *out_token)
        {
            (void)draft_sample_slot;
            (void)out_token;
            return false;
        }

        /**
         * @brief Sample one row from all-position verifier logits in greedy mode.
         *
         * @param row Logical verifier row to sample.
         * @return Token id, or -1 when unavailable.
         */
        virtual int sampleGreedyFromAllPositionLogitsOnDevice(int row)
        {
            (void)row;
            return -1;
        }

        /**
         * @brief Sample several contiguous verifier-logit rows in greedy mode.
         *
         * Implementations may use a backend batched argmax to avoid one
         * host/device synchronization per row. The default preserves the
         * existing contract by sampling each row individually, with a host-side
         * greedy scan fallback when all-position logits are already CPU-visible.
         */
        virtual bool sampleGreedyFromAllPositionLogitsOnDeviceRows(
            int start_row,
            int row_count,
            int32_t *out_tokens)
        {
            if (start_row < 0 || row_count <= 0 || !out_tokens)
                return false;

            const float *all_logits = getAllPositionLogits();
            const int vocab = vocab_size();
            for (int i = 0; i < row_count; ++i)
            {
                const int row = start_row + i;
                int token = sampleGreedyFromAllPositionLogitsOnDevice(row);
                if (token < 0 && all_logits && vocab > 0)
                {
                    const float *row_logits =
                        all_logits + static_cast<size_t>(row) * static_cast<size_t>(vocab);
                    int best = 0;
                    float best_value = row_logits[0];
                    for (int col = 1; col < vocab; ++col)
                    {
                        if (row_logits[col] > best_value)
                        {
                            best_value = row_logits[col];
                            best = col;
                        }
                    }
                    token = best;
                }
                if (token < 0)
                    return false;
                out_tokens[i] = static_cast<int32_t>(token);
            }
            return true;
        }

        /**
         * @brief True when greedy all-position verifier rows can be reduced into
         *        a compact speculative-verify outcome on the producing device.
         *
         * This is stricter than "can sample verifier rows".  LocalTP can sample
         * sharded verifier rows by gathering child-local argmax results at the
         * rank level, but it cannot yet run one device-side reducer over all TP
         * shards.  Callers must check this capability before invoking
         * verifyGreedyAllPositionBatchOutcomeOnDevice().
         */
        virtual bool supportsGreedyAllPositionBatchOutcomeOnDevice() const
        {
            return false;
        }

        /**
         * @brief Summarize greedy all-position MTP verifier rows on device.
         *
         * This is the greedy counterpart to
         * verifyStochasticDistributionsBatchOutcomeOnDevice(): implementations
         * should sample verifier logits on the graph replay stream, compare the
         * device-resident verifier tokens with the device-resident compact
         * verifier input row, and return only the already-reduced vLLM-style
         * commit outcome. The default returns false so unsupported topologies
         * keep using the older host-row path.
         */
        virtual bool verifyGreedyAllPositionBatchOutcomeOnDevice(
            const int32_t *draft_tokens,
            int draft_token_count,
            const int32_t *stop_tokens,
            int stop_token_count,
            DeviceSpeculativeVerifyBatchOutcome *out)
        {
            (void)draft_tokens;
            (void)draft_token_count;
            (void)stop_tokens;
            (void)stop_token_count;
            (void)out;
            return false;
        }

        /**
         * @brief Summarize greedy all-position verifier rows and keep the
         *        compact outcome device-resident.
         *
         * This is the GPU hot-path counterpart to
         * verifyGreedyAllPositionBatchOutcomeOnDevice().  Implementations
         * enqueue verifier-row argmax plus compact accepted-token metadata on
         * an explicit producer stream and return a DeviceSpeculativeOutcomeHandle
         * that can be consumed by device-resident publication before any host
         * response bridge.  The legacy host-visible verifier may delegate to
         * this method and then call materializeDeviceSpeculativeOutcomesForHostResponse().
         */
        virtual bool verifyGreedyAllPositionBatchOutcomeOnDeviceResident(
            const int32_t *draft_tokens,
            int draft_token_count,
            const int32_t *stop_tokens,
            int stop_token_count,
            DeviceSpeculativeOutcomeHandle *out_handle)
        {
            (void)draft_tokens;
            (void)draft_token_count;
            (void)stop_tokens;
            (void)stop_token_count;
            (void)out_handle;
            return false;
        }

        /**
         * @brief Batched forward pass
         *
         * Process multiple sequences in parallel with automatic padding.
         *
         * @param token_batches Vector of token sequences
         * @return true if forward pass succeeded
         */
        virtual bool forward_batch(const std::vector<std::vector<int>> &token_batches)
        {
            (void)token_batches;
            return false; // Default: not implemented
        }

        /**
         * @brief Whether this runner can initialize a benchmark request batch.
         *
         * Request-batched MTP decode requires every logical request slot to own
         * valid prompt KV/state before decode begins. BenchmarkRunner checks
         * this capability before using `prefillBatchForBenchmark()` so a
         * request-batched lane cannot accidentally measure request 0 only.
         */
        virtual bool supportsPrefillBatchForBenchmark(int request_batch) const
        {
            (void)request_batch;
            return false;
        }

        /**
         * @brief Initialize a benchmark request batch with padded prefill.
         *
         * `token_batches[i]` is the prompt for logical request `i`. The default
         * implementation is unsupported because high-level request ownership
         * must opt in explicitly.
         */
        virtual bool prefillBatchForBenchmark(
            const std::vector<std::vector<int>> &token_batches)
        {
            (void)token_batches;
            return false;
        }

        /**
         * @brief Get logits for a specific sequence in batch
         *
         * Returns logits for the specified sequence index.
         * For E2E tests that compare all positions.
         *
         * @param seq_idx Sequence index in batch (default=0)
         * @return Pointer to logits [padded_seq_len, vocab_size], or nullptr
         */
        virtual const float *getLogits(int seq_idx = 0) const
        {
            (void)seq_idx;
            return logits(); // Default: return single-sequence logits
        }

        /**
         * @brief Get current batch size
         */
        virtual int batch_size() const { return 1; }

        /**
         * @brief Get padded sequence length for current batch
         */
        virtual int padded_seq_len() const { return 0; }

        /**
         * @brief Get sequence lengths for current batch
         *
         * @return Vector of actual (unpadded) sequence lengths
         */
        virtual const std::vector<int> &sequence_lengths() const
        {
            static const std::vector<int> empty;
            return empty;
        }

        /**
         * @brief Return pending device-resident logical state, if any.
         *
         * This is the typed no-D2H handoff for Phase 10 resident MTP
         * publication.  The default implementation is empty because most
         * runners still expose only host-owned positions and sequence lengths.
         */
        virtual DeviceResidentLogicalSequenceStateHandle deviceResidentLogicalSequenceState() const
        {
            return {};
        }

        /**
         * @brief Return whether host logical getters mirror any resident mailbox.
         *
         * A valid deviceResidentLogicalSequenceState() means a runner has
         * staged logical positions/sequence lengths on device. Until the
         * compatibility host bridge adopts the equivalent step plan,
         * get_position() and sequence_lengths() may be stale and must not drive
         * speculative planning. Runners without resident state can keep the
         * default true result.
         */
        virtual bool hostLogicalStateMirrorsDeviceResidentState() const
        {
            return true;
        }

        /**
         * @brief Get vocabulary size
         */
        virtual int vocab_size() const = 0;

        /**
         * @brief Reset request-scoped inference state before a new prompt/session.
         *
         * Despite the historical name, this is not a graph-cache teardown API.
         * Callers use it at request boundaries, benchmark iteration boundaries,
         * and after hard replay/restore failures when the next token stream must
         * start from an empty live sequence.
         *
         * Implementations must clear all live sequence state:
         * - main and MTP KV cache contents;
         * - recurrent/hybrid model state such as GDN or short-conv buffers;
         * - host/device logical positions and sequence lengths;
         * - pending sampled-token handoffs, streams, and request-local metadata.
         *
         * Implementations must preserve long-lived execution assets:
         * - reusable ComputeGraph topology and captured graph objects;
         * - BufferArena registrations, workspace bindings, and prepared weights;
         * - device contexts and backend-owned model allocations.
         *
         * A future destructive topology/workspace reset should use a different,
         * explicitly named API.  Do not overload clear_cache() for that purpose.
         */
        virtual void clear_cache() = 0;

        /**
         * @brief GPU-side greedy sampling (skip D2H of full logits)
         *
         * For multi-GPU TP inference, this performs argmax on each device's
         * local logits partition on the GPU, then D2H only the (value, index)
         * result pairs (8 bytes per device vs ~600 KB for full logits).
         *
         * @return Token ID (>= 0) if on-device sampling succeeded,
         *         -1 if not supported (caller should fall back to logits() + CPU argmax)
         */
        virtual int sampleGreedyOnDevice() { return -1; }

        /**
         * @brief Sample current main logits greedily into a device target slot.
         *
         * GPU MTP verification uses target-token slots as the source of truth
         * for deferred first-token handoff.  Implementations enqueue a
         * graph-capturable argmax on an explicit stream and write the selected
         * token into the same runner-owned target sample arena consumed by
         * prepareMTPVerifierInputTokensOnDeviceFromDeviceFirstToken() and
         * forwardMTPFromDeviceTargetForDeviceSampling().  Passing nullptr for
         * @p out_token requests a fully deferred sample with no D2H copy.
         */
        virtual bool sampleGreedyFromMainLogitsToDeviceTargetSlot(
            int target_sample_slot,
            int32_t *out_token)
        {
            (void)target_sample_slot;
            (void)out_token;
            return false;
        }

        /**
         * @brief GPU-side sampling with full top-k/top-p support
         *
         * For greedy (temperature=0), delegates to sampleGreedyOnDevice().
         * For non-greedy, runs per-device GPU top-k selection, then performs
         * cross-device merge + softmax + top-p filtering + sampling on host
         * (operating on only k candidates, not the full vocabulary).
         *
         * @param params Sampling parameters (temperature, top_k, top_p, seed)
         * @return Token ID (>= 0) on success, -1 if not supported
         */
        virtual int sampleOnDevice(const SamplingParams &params)
        {
            (void)params;
            return -1;
        }

        /**
         * @brief Whether MPI worker ranks must enter decode sampling with rank 0.
         *
         * Some runners sample from already-gathered logits or fall back to a
         * root-local CPU sampler.  In those cases worker ranks must not call
         * sampleGreedyOnDevice()/sampleOnDevice(), because there is no matching
         * collective on rank 0 and they can deadlock.  Vocab-sharded/global-TP
         * runners that coordinate sampling candidates across ranks should return
         * true for the matching SamplingParams so the server command loop keeps
         * every rank in the sampling collective before rank 0 publishes the
         * authoritative token.
         */
        virtual bool requiresMPICoordinatedDecodeSampling(const SamplingParams &params) const
        {
            (void)params;
            return false;
        }

        /**
         * @brief Sample request-batched main logits that already live on device.
         *
         * request-batched prefill writes one terminal logits row per logical
         * request. GPU runners must sample those rows through this hook instead
         * of exposing a device pointer to the CPU Sampler. Implementations own
         * stream ordering and any compact D2H copy of selected token ids.
         *
         * @param request_count Number of active request rows to sample.
         * @param params Sampling parameters for all rows.
         * @param out_tokens Host output buffer [request_count].
         * @param stochastic_thresholds Optional host thresholds
         *        [request_count] for non-greedy sampling.  vLLM-style MTP
         *        keys stochastic draws by logical output position, so callers
         *        that request stochastic device sampling must provide the
         *        exact per-request thresholds instead of letting the runner
         *        advance a backend-local RNG counter.
         * @return true when every row was sampled on the runner device.
         */
        virtual bool sampleMainLogitsBatchRowsOnDevice(
            int request_count,
            const SamplingParams &params,
            int32_t *out_tokens,
            const float *stochastic_thresholds = nullptr)
        {
            (void)request_count;
            (void)params;
            (void)out_tokens;
            (void)stochastic_thresholds;
            return false;
        }

        /**
         * @brief Whether this runner can execute a full high-level decode step.
         */
        virtual bool supportsDecodeStep() const { return false; }

        /**
         * @brief Whether this runner can execute a full batched decode step.
         *
         * This is intentionally separate from supportsDecodeStep(): request
         * batching has different ownership and publication semantics, and a
         * runner must opt in explicitly before benchmarks may report batched
         * MTP throughput.
         *
         * @param request_batch Number of logical requests in the batch.
         */
        virtual bool supportsDecodeStepBatchForBenchmark(int request_batch) const
        {
            (void)request_batch;
            return false;
        }

        /**
         * @brief Set sampling params consumed by decodeStepForBenchmark().
         */
        virtual void setDecodeSamplingParams(const SamplingParams & /*params*/) {}

        /**
         * @brief Limit how many tokens the next decode step may accept.
         */
        virtual void setDecodeStepTokenBudget(int /*max_tokens*/) {}

        /**
         * @brief Execute one high-level decode step.
         */
        virtual DecodeStepOutput decodeStepForBenchmark()
        {
            return DecodeStepOutput{{}, false, "decodeStepForBenchmark unsupported"};
        }

        /**
         * @brief Execute one high-level batched decode step.
         *
         * The caller supplies a per-request token budget with
         * setDecodeStepTokenBudget() before this call. Implementations must not
         * emit more than that many tokens for any single request.
         *
         * @param request_batch Number of active logical requests.
         */
        virtual DecodeBatchStepOutput decodeBatchStepForBenchmark(int request_batch)
        {
            (void)request_batch;
            return DecodeBatchStepOutput{{}, {}, "decodeBatchStepForBenchmark unsupported"};
        }

        /**
         * @brief Apply decode-boundary maintenance after a successful step.
         */
        virtual bool maybeApplyDecodeBoundaryMaintenance() { return true; }

        /**
         * @brief Apply sparse logit penalties on device (GPU-side)
         *
         * Uploads a sparse penalty map to the GPU and applies it in-place to the
         * logits tensor. This avoids a full D2H transfer of the logits tensor
         * (~600KB for 151K vocab) just to apply penalties.
         *
         * After this call, the penalized logits remain on GPU and can be sampled
         * via sampleGreedyOnDevice() or sampleOnDevice().
         *
         * @param penalties Sparse penalty entries (token_id, penalty) to subtract
         * @param vocab_size Vocabulary size (for bounds checking)
         * @return true if applied on device, false if not supported (caller should fall back)
         */
        virtual bool applyPenaltiesOnDevice(const std::vector<LogitPenalty> &penalties,
                                            int vocab_size)
        {
            (void)penalties;
            (void)vocab_size;
            return false;
        }

        /**
         * @brief Apply sparse logit penalties to MTP sidecar logits on device.
         */
        virtual bool applyPenaltiesToMTPLogitsOnDevice(const std::vector<LogitPenalty> &penalties,
                                                       int vocab_size)
        {
            (void)penalties;
            (void)vocab_size;
            return false;
        }

        /**
         * @brief Apply sparse logit penalties to one all-position verifier row on device.
         */
        virtual bool applyPenaltiesToAllPositionLogitsOnDeviceRow(
            int row,
            const std::vector<LogitPenalty> &penalties,
            int vocab_size)
        {
            (void)row;
            (void)penalties;
            (void)vocab_size;
            return false;
        }

        /**
         * @brief True when all-position verifier rows can receive branch-local
         *        sampler penalties before compact verifier outcome reduction.
         *
         * Greedy sampling with repetition/frequency/DRY penalties is still an
         * argmax, but every verifier row observes a different speculative
         * history: first emitted token, then only the previously accepted draft
         * rows.  Backends should advertise this only when
         * applyPenaltiesToAllPositionLogitsOnDeviceRow() mutates the producer
         * row on an explicit stream without forcing a full logits readback.
         */
        virtual bool supportsRowLocalAllPositionPenaltyApplication() const
        {
            return false;
        }

        /**
         * @brief True when compact device-side stochastic MTP distributions are available.
         *
         * This is intentionally narrower than generic sampling support: it means
         * the runner can build compact top-k/top-p probability tables for main,
         * MTP, and all-position verifier logits using explicit streams and
         * arena-owned buffers, then verify accept/residual decisions without
         * copying full logits to host.
         */
        virtual bool supportsDeviceStochasticMTPVerification() const { return false; }

        virtual bool buildStochasticDistributionOnDevice(
            DeviceLogitsSource source,
            int row,
            DeviceDistributionBuffer buffer,
            int slot,
            const SamplingParams &params,
            int vocab_size)
        {
            (void)source;
            (void)row;
            (void)buffer;
            (void)slot;
            (void)params;
            (void)vocab_size;
            return false;
        }

        /**
         * @brief Build compact top-k/top-p tables for contiguous verifier rows.
         *
         * The default implementation is unsupported. Device graph runners use
         * this to queue all all-position target/bonus verifier rows on the
         * verifier replay stream, avoiding one scalar distribution launch pair
         * per row. Implementations must require an explicit stream and must not
         * synchronize.
         */
        virtual bool buildStochasticDistributionsOnDevice(
            DeviceLogitsSource source,
            int first_row,
            DeviceDistributionBuffer buffer,
            int first_slot,
            int row_count,
            const SamplingParams &params,
            int vocab_size)
        {
            (void)source;
            (void)first_row;
            (void)buffer;
            (void)first_slot;
            (void)row_count;
            (void)params;
            (void)vocab_size;
            return false;
        }

        /**
         * @brief Legacy full-vocab stochastic probability row builder.
         *
         * Production vLLM-style greedy-draft MTP now prefers compact
         * buildStochasticDistributionsOnDevice() rows plus the one-hot batched
         * outcome verifier. Full target/draft probability rows are intentionally
         * not part of the production GPU runner contract because they allocate
         * extra vocab-sized scratch and recreate the removed scalar verifier
         * path. This default remains unsupported for older tests and
         * non-production runners.
         */
        virtual bool buildStochasticProbabilityRowsOnDevice(
            DeviceLogitsSource source,
            int first_row,
            DeviceDistributionBuffer buffer,
            int first_slot,
            int row_count,
            const SamplingParams &params,
            int vocab_size)
        {
            (void)source;
            (void)first_row;
            (void)buffer;
            (void)first_slot;
            (void)row_count;
            (void)params;
            (void)vocab_size;
            return false;
        }

        /**
         * @brief Build processed full-logit rows without softmax materialization.
         *
         * The rows are in sampling space after temperature, top-k/top-p masks,
         * and penalties. They remain useful for full-probability parity tests
         * and future non-greedy draft experiments, but production greedy-draft
         * MTP should use compact distribution rows instead.
         */
        virtual bool buildStochasticProcessedLogitRowsOnDevice(
            DeviceLogitsSource source,
            int first_row,
            DeviceDistributionBuffer buffer,
            int first_slot,
            int row_count,
            const SamplingParams &params,
            int vocab_size)
        {
            (void)source;
            (void)first_row;
            (void)buffer;
            (void)first_slot;
            (void)row_count;
            (void)params;
            (void)vocab_size;
            return false;
        }

        /**
         * @brief Build and sample a vLLM-style MTP draft proposal on device.
         *
         * Draft proposal follows vLLM's default greedy draft branch: the
         * runner stores the sampled draft token and q(sampled_token) in
         * runner-owned device slots, and later verifier work treats q as
         * one-hot at that draft token.
         */
        virtual int sampleStochasticDraftProposalOnDevice(
            DeviceLogitsSource source,
            int row,
            int slot,
            const SamplingParams &params,
            int vocab_size,
            float threshold)
        {
            (void)source;
            (void)row;
            (void)slot;
            (void)params;
            (void)vocab_size;
            (void)threshold;
            return -1;
        }

        /**
         * @brief Deferred-host-read variant of sampleStochasticDraftProposalOnDevice().
         *
         * The sampled token stays in the runner-owned device draft sample slot,
         * with an explicit readiness event recorded for the verifier stream.
         */
        virtual bool sampleStochasticDraftProposalOnDeviceDeferred(
            DeviceLogitsSource source,
            int row,
            int slot,
            const SamplingParams &params,
            int vocab_size,
            float threshold)
        {
            (void)source;
            (void)row;
            (void)slot;
            (void)params;
            (void)vocab_size;
            (void)threshold;
            return false;
        }

        /**
         * @brief Stage sampled draft tokens into verifier-owned device slots.
         *
         * Request-batched stochastic verification amortizes the target
         * verifier forward across requests.  This hook gives the runner a
         * structured, explicit-stream way to copy one request's already-sampled
         * greedy draft tokens into the caller-selected device slots consumed by
         * verifyStochasticDistributionsBatchOutcomeOnDevice().
         *
         * The default hard-fails. GPU implementations must use an explicit
         * non-null stream and record the normal draft-sample readiness events.
         */
        virtual bool stageStochasticDraftTokensForDeviceVerification(
            const int32_t *draft_tokens,
            int draft_token_count,
            int first_draft_slot = 0)
        {
            (void)draft_tokens;
            (void)draft_token_count;
            (void)first_draft_slot;
            return false;
        }

        virtual int sampleStochasticDistributionOnDevice(
            DeviceDistributionBuffer buffer,
            int slot,
            float threshold)
        {
            (void)buffer;
            (void)slot;
            (void)threshold;
            return -1;
        }

        /**
         * @brief Sample a compact stochastic distribution into runner-owned device memory.
         *
         * This is the no-host-read companion to sampleStochasticDistributionOnDevice().
         * It is used when later graph stages consume the sampled token directly
         * from the runner's device sample slot.  The caller must not require the
         * sampled scalar on the host before the batch outcome is summarized.
         *
         * @param buffer Target or draft distribution buffer to sample from.
         * @param slot Compact distribution slot.
         * @param threshold Pre-drawn uniform threshold for deterministic replay.
         * @return true if the sample kernel was enqueued and wrote the device slot.
         */
        virtual bool sampleStochasticDistributionOnDeviceDeferred(
            DeviceDistributionBuffer buffer,
            int slot,
            float threshold)
        {
            (void)buffer;
            (void)slot;
            (void)threshold;
            return false;
        }

        /**
         * @brief Compose verifier input tokens when the first token is on device.
         *
         * The returned pointer names a stable runner-owned INT32 row with shape
         * `[total_verifier_input_tokens]`. Entry zero is copied from the target
         * sample slot and entries one..N are copied from sampled draft slots on
         * the graph execution stream.
         */
        virtual const void *prepareMTPVerifierInputTokensOnDeviceFromDeviceFirstToken(
            int first_target_sample_slot,
            int first_draft_slot,
            int draft_token_count,
            int total_verifier_input_tokens)
        {
            (void)first_target_sample_slot;
            (void)first_draft_slot;
            (void)draft_token_count;
            (void)total_verifier_input_tokens;
            return nullptr;
        }

        virtual bool verifyStochasticDistributionsOnDevice(
            int target_slot,
            int draft_slot,
            int draft_token,
            float accept_threshold,
            float residual_threshold,
            DeviceSpeculativeVerifyResult *out)
        {
            (void)target_slot;
            (void)draft_slot;
            (void)draft_token;
            (void)accept_threshold;
            (void)residual_threshold;
            (void)out;
            return false;
        }

        /**
         * @brief Verify one stochastic MTP row using full probability rows.
         *
         * This is the row-level vLLM-style path for history-dependent sampling
         * where penalties require target rows to be processed sequentially.  The
         * sampled draft token must already live in the runner's draft device
         * slot; `draft_token` remains a host-side metadata shadow for tests and
         * diagnostics.
         */
        /**
         * @brief Legacy scalar full-probability stochastic verifier.
         *
         * Production GPU runners should prefer
         * verifyStochasticDistributionsBatchOutcomeOnDevice() with
         * use_vllm_probability_rejection=true. The default false implementation
         * prevents silent fallback to an unowned full-probability arena path.
         */
        virtual bool verifyStochasticProbabilityRowOnDevice(
            int target_slot,
            int draft_slot,
            int draft_token,
            float accept_threshold,
            uint64_t inverse_sample_seed,
            int inverse_sample_logical_position,
            DeviceSpeculativeVerifyResult *out)
        {
            (void)target_slot;
            (void)draft_slot;
            (void)draft_token;
            (void)accept_threshold;
            (void)inverse_sample_seed;
            (void)inverse_sample_logical_position;
            (void)out;
            return false;
        }

        virtual bool verifyStochasticDistributionsBatchOnDevice(
            int first_target_slot,
            int first_draft_slot,
            const int32_t *draft_tokens,
            const float *accept_thresholds,
            const float *residual_thresholds,
            int row_count,
            DeviceSpeculativeVerifyResult *out)
        {
            (void)first_target_slot;
            (void)first_draft_slot;
            (void)draft_tokens;
            (void)accept_thresholds;
            (void)residual_thresholds;
            (void)row_count;
            (void)out;
            return false;
        }

        /**
         * @brief Verify and summarize several stochastic MTP rows on device.
         *
         * This is the vLLM-style output contract for penalty-free stochastic
         * all-position verification: backend kernels decide row acceptance,
         * reduce the first consumed rejection/stop/all-accepted outcome, and
         * return only the committed token sequence plus counters to the host.
         * `draft_tokens` may be null for runners that keep sampled draft tokens
         * in device slots; those implementations must use `first_draft_slot`
         * as the source of truth instead of a host token shadow.
         */
        virtual bool verifyStochasticDistributionsBatchOutcomeOnDevice(
            int first_target_slot,
            int first_draft_slot,
            const int32_t *draft_tokens,
            const float *accept_thresholds,
            const float *residual_thresholds,
            int row_count,
            int32_t first_token,
            const int32_t *stop_tokens,
            int stop_token_count,
            int bonus_target_slot,
            float bonus_threshold,
            DeviceSpeculativeVerifyBatchOutcome *out,
            uint64_t inverse_sample_seed = 0,
            int inverse_sample_first_logical_position = 0,
            bool use_vllm_probability_rejection = false)
        {
            (void)first_target_slot;
            (void)first_draft_slot;
            (void)draft_tokens;
            (void)accept_thresholds;
            (void)residual_thresholds;
            (void)row_count;
            (void)first_token;
            (void)stop_tokens;
            (void)stop_token_count;
            (void)bonus_target_slot;
            (void)bonus_threshold;
            (void)out;
            (void)inverse_sample_seed;
            (void)inverse_sample_first_logical_position;
            (void)use_vllm_probability_rejection;
            return false;
        }

        /**
         * @brief Device-first-token form of batched stochastic verification.
         *
         * This keeps the initial main-model sample on GPU until the summary
         * kernel has decided which tokens commit. It should be used only for
         * penalty-free paths where sampler history does not need the first
         * token before verifier reduction. `draft_tokens` follows the same
         * nullable device-slot contract as the host-first overload.
         */
        virtual bool verifyStochasticDistributionsBatchOutcomeOnDeviceFirstToken(
            int first_target_slot,
            int first_draft_slot,
            const int32_t *draft_tokens,
            const float *accept_thresholds,
            const float *residual_thresholds,
            int row_count,
            int first_target_sample_slot,
            const int32_t *stop_tokens,
            int stop_token_count,
            int bonus_target_slot,
            float bonus_threshold,
            DeviceSpeculativeVerifyBatchOutcome *out,
            uint64_t inverse_sample_seed = 0,
            int inverse_sample_first_logical_position = 0,
            bool use_vllm_probability_rejection = false)
        {
            (void)first_target_slot;
            (void)first_draft_slot;
            (void)draft_tokens;
            (void)accept_thresholds;
            (void)residual_thresholds;
            (void)row_count;
            (void)first_target_sample_slot;
            (void)stop_tokens;
            (void)stop_token_count;
            (void)bonus_target_slot;
            (void)bonus_threshold;
            (void)out;
            (void)inverse_sample_seed;
            (void)inverse_sample_first_logical_position;
            (void)use_vllm_probability_rejection;
            return false;
        }

        /**
         * @brief Enqueue a single stochastic MTP outcome and keep it device-resident.
         *
         * This scalar convenience wrapper preserves the legacy call shape while
         * routing through the same request-batch resident contract used by the
         * scheduler-oriented path.  Future publication code can consume the
         * returned handle directly; compatibility callers should immediately
         * bridge it with materializeDeviceSpeculativeOutcomesForHostResponse().
         */
        virtual bool verifyStochasticDistributionsBatchOutcomeOnDeviceResident(
            int first_target_slot,
            int first_draft_slot,
            const int32_t *draft_tokens,
            const float *accept_thresholds,
            const float *residual_thresholds,
            int row_count,
            int32_t first_token,
            const int32_t *stop_tokens,
            int stop_token_count,
            int bonus_target_slot,
            float bonus_threshold,
            DeviceSpeculativeOutcomeHandle *out_handle,
            uint64_t inverse_sample_seed = 0,
            int inverse_sample_first_logical_position = 0,
            bool use_vllm_probability_rejection = false)
        {
            using namespace sampling_math;
            const bool derive_thresholds_from_seed =
                accept_thresholds == nullptr &&
                residual_thresholds == nullptr &&
                use_vllm_probability_rejection &&
                inverse_sample_seed != 0 &&
                inverse_sample_first_logical_position >= 0;
            const bool has_host_thresholds =
                accept_thresholds != nullptr && residual_thresholds != nullptr;
            if ((!has_host_thresholds && !derive_thresholds_from_seed) ||
                row_count <= 0 ||
                row_count > kSpeculativeBatchMaxRows ||
                stop_token_count < 0 ||
                stop_token_count > kSpeculativeBatchMaxStopTokens ||
                (stop_token_count > 0 && !stop_tokens) ||
                !out_handle)
            {
                return false;
            }

            DeviceStochasticBatchOutcomeRequest request;
            request.request_id = 0;
            request.first_target_slot = first_target_slot;
            request.first_draft_slot = first_draft_slot;
            request.row_count = row_count;
            request.first_token = first_token;
            request.first_token_from_device = false;
            request.bonus_target_slot = bonus_target_slot;
            request.bonus_threshold = bonus_threshold;
            request.inverse_sample_seed = inverse_sample_seed;
            request.inverse_sample_first_logical_position =
                inverse_sample_first_logical_position;
            request.use_vllm_probability_rejection =
                use_vllm_probability_rejection;
            request.derive_thresholds_from_seed = derive_thresholds_from_seed;
            request.use_device_draft_tokens = draft_tokens == nullptr;

            for (int row = 0; row < row_count; ++row)
            {
                if (has_host_thresholds)
                {
                    request.accept_thresholds[static_cast<size_t>(row)] =
                        accept_thresholds[row];
                    request.residual_thresholds[static_cast<size_t>(row)] =
                        residual_thresholds[row];
                }
                if (draft_tokens)
                {
                    request.draft_tokens[static_cast<size_t>(row)] =
                        draft_tokens[row];
                }
            }
            request.stop_token_count = stop_token_count;
            for (int i = 0; i < stop_token_count; ++i)
            {
                request.stop_tokens[static_cast<size_t>(i)] = stop_tokens[i];
            }

            return verifyStochasticDistributionsRequestBatchOutcomesOnDeviceResident(
                &request,
                /*request_count=*/1,
                out_handle);
        }

        /**
         * @brief Device-first-token resident variant of scalar stochastic MTP outcome.
         *
         * The first main-model token remains in a device sample slot.  The
         * compact summary kernel reads it on the verifier stream and leaves the
         * accepted-count/output metadata device-resident.
         */
        virtual bool verifyStochasticDistributionsBatchOutcomeOnDeviceFirstTokenResident(
            int first_target_slot,
            int first_draft_slot,
            const int32_t *draft_tokens,
            const float *accept_thresholds,
            const float *residual_thresholds,
            int row_count,
            int first_target_sample_slot,
            const int32_t *stop_tokens,
            int stop_token_count,
            int bonus_target_slot,
            float bonus_threshold,
            DeviceSpeculativeOutcomeHandle *out_handle,
            uint64_t inverse_sample_seed = 0,
            int inverse_sample_first_logical_position = 0,
            bool use_vllm_probability_rejection = false)
        {
            using namespace sampling_math;
            const bool derive_thresholds_from_seed =
                accept_thresholds == nullptr &&
                residual_thresholds == nullptr &&
                use_vllm_probability_rejection &&
                inverse_sample_seed != 0 &&
                inverse_sample_first_logical_position >= 0;
            const bool has_host_thresholds =
                accept_thresholds != nullptr && residual_thresholds != nullptr;
            if ((!has_host_thresholds && !derive_thresholds_from_seed) ||
                row_count <= 0 ||
                row_count > kSpeculativeBatchMaxRows ||
                first_target_sample_slot < 0 ||
                stop_token_count < 0 ||
                stop_token_count > kSpeculativeBatchMaxStopTokens ||
                (stop_token_count > 0 && !stop_tokens) ||
                !out_handle)
            {
                return false;
            }

            DeviceStochasticBatchOutcomeRequest request;
            request.request_id = 0;
            request.first_target_slot = first_target_slot;
            request.first_draft_slot = first_draft_slot;
            request.row_count = row_count;
            request.first_token = -1;
            request.first_token_from_device = true;
            request.first_target_sample_slot = first_target_sample_slot;
            request.bonus_target_slot = bonus_target_slot;
            request.bonus_threshold = bonus_threshold;
            request.inverse_sample_seed = inverse_sample_seed;
            request.inverse_sample_first_logical_position =
                inverse_sample_first_logical_position;
            request.use_vllm_probability_rejection =
                use_vllm_probability_rejection;
            request.derive_thresholds_from_seed = derive_thresholds_from_seed;
            request.use_device_draft_tokens = draft_tokens == nullptr;

            for (int row = 0; row < row_count; ++row)
            {
                if (has_host_thresholds)
                {
                    request.accept_thresholds[static_cast<size_t>(row)] =
                        accept_thresholds[row];
                    request.residual_thresholds[static_cast<size_t>(row)] =
                        residual_thresholds[row];
                }
                if (draft_tokens)
                {
                    request.draft_tokens[static_cast<size_t>(row)] =
                        draft_tokens[row];
                }
            }
            request.stop_token_count = stop_token_count;
            for (int i = 0; i < stop_token_count; ++i)
            {
                request.stop_tokens[static_cast<size_t>(i)] = stop_tokens[i];
            }

            return verifyStochasticDistributionsRequestBatchOutcomesOnDeviceResident(
                &request,
                /*request_count=*/1,
                out_handle);
        }

        /**
         * @brief Verify and summarize a logical request batch in one runner call.
         *
         * This is the orchestration-level contract for vLLM-style stochastic
         * request batching. Callers must stage all target rows, draft slots, and
         * RNG draws first, then hand the complete descriptor list to the runner.
         * The default implementation is intentionally strict but conservative:
         * it validates the descriptors and delegates each request to the
         * existing single-request virtual reducer. CUDA/ROCm runners can
         * override this with a true multi-output backend kernel while preserving
         * the same public contract.
         */
        virtual bool verifyStochasticDistributionsRequestBatchOutcomesOnDevice(
            const DeviceStochasticBatchOutcomeRequest *requests,
            int request_count,
            DeviceSpeculativeVerifyBatchOutcome *outcomes)
        {
            using namespace sampling_math;
            if (!requests || !outcomes || request_count <= 0)
                return false;

            for (int i = 0; i < request_count; ++i)
            {
                const DeviceStochasticBatchOutcomeRequest &request = requests[i];
                if (request.row_count <= 0 ||
                    request.row_count > kSpeculativeBatchMaxRows ||
                    request.stop_token_count < 0 ||
                    request.stop_token_count > kSpeculativeBatchMaxStopTokens)
                {
                    return false;
                }

                const int32_t *stop_tokens =
                    request.stop_token_count > 0
                        ? request.stop_tokens.data()
                        : nullptr;
                const int32_t *draft_tokens =
                    request.hostDraftTokensOrNull();
                std::array<float, kSpeculativeBatchMaxRows>
                    derived_accept_thresholds = {};
                std::array<float, kSpeculativeBatchMaxRows>
                    derived_residual_thresholds = {};
                const float *accept_thresholds =
                    request.accept_thresholds.data();
                const float *residual_thresholds =
                    request.residual_thresholds.data();
                if (request.derive_thresholds_from_seed)
                {
                    for (int row = 0; row < request.row_count; ++row)
                    {
                        const int logical_position =
                            request.inverse_sample_first_logical_position + row;
                        derived_accept_thresholds[static_cast<size_t>(row)] =
                            mtp_spec_threshold_from_seed(
                                request.inverse_sample_seed,
                                logical_position,
                                1 /* MTPSpecStochasticDrawPurpose::Accept */);
                        derived_residual_thresholds[static_cast<size_t>(row)] =
                            mtp_spec_threshold_from_seed(
                                request.inverse_sample_seed,
                                logical_position,
                                2 /* MTPSpecStochasticDrawPurpose::Residual */);
                    }
                    accept_thresholds = derived_accept_thresholds.data();
                    residual_thresholds = derived_residual_thresholds.data();
                }

                bool ok = false;
                if (request.first_token_from_device)
                {
                    ok = verifyStochasticDistributionsBatchOutcomeOnDeviceFirstToken(
                        request.first_target_slot,
                        request.first_draft_slot,
                        draft_tokens,
                        accept_thresholds,
                        residual_thresholds,
                        request.row_count,
                        request.first_target_sample_slot,
                        stop_tokens,
                        request.stop_token_count,
                        request.bonus_target_slot,
                        request.bonus_threshold,
                        outcomes + i,
                        request.inverse_sample_seed,
                        request.inverse_sample_first_logical_position,
                        request.use_vllm_probability_rejection);
                }
                else
                {
                    ok = verifyStochasticDistributionsBatchOutcomeOnDevice(
                        request.first_target_slot,
                        request.first_draft_slot,
                        draft_tokens,
                        accept_thresholds,
                        residual_thresholds,
                        request.row_count,
                        request.first_token,
                        stop_tokens,
                        request.stop_token_count,
                        request.bonus_target_slot,
                        request.bonus_threshold,
                        outcomes + i,
                        request.inverse_sample_seed,
                        request.inverse_sample_first_logical_position,
                        request.use_vllm_probability_rejection);
                }
                if (!ok)
                    return false;
            }

            return true;
        }

        /**
         * @brief Enqueue request-batch verification and leave compact outcome on device.
         *
         * Implementations should enqueue all per-request verify/bonus/summary
         * kernels on one explicit stream and return a handle to compact device
         * output rows.  This is the GPU-resident Phase 10 contract; callers that
         * still need host-visible response metadata should call
         * materializeDeviceSpeculativeOutcomesForHostResponse().
         */
        virtual bool verifyStochasticDistributionsRequestBatchOutcomesOnDeviceResident(
            const DeviceStochasticBatchOutcomeRequest *requests,
            int request_count,
            DeviceSpeculativeOutcomeHandle *out_handle)
        {
            (void)requests;
            (void)request_count;
            (void)out_handle;
            return false;
        }

        /**
         * @brief Compatibility bridge from a device-resident outcome to host structs.
         *
         * This method intentionally represents the legacy host-visible boundary.
         * New state-publication code should consume DeviceSpeculativeOutcomeHandle
         * directly instead of calling this bridge in the decode hot path.
         */
        virtual bool copyDeviceSpeculativeOutcomesToHost(
            const DeviceSpeculativeOutcomeHandle &handle,
            DeviceSpeculativeVerifyBatchOutcome *outcomes)
        {
            (void)handle;
            (void)outcomes;
            return false;
        }

        /**
         * @brief Materialize resident stochastic outcomes for legacy host plans.
         *
         * Prefer materializeDeviceSpeculativeOutcomesForHostResponse() for new
         * code.  This compatibility entry point remains for older tests and
         * host-plan adapters that have not yet been split into response-only and
         * diagnostic consumers.
         *
         * The default delegates to the historical host-copy hook so existing
         * runners and tests keep one implementation.  Overrides should preserve
         * the same ordering: wait on `handle.response_ready_event` from an
         * explicit response bridge stream, enqueue compact D2H copies on that
         * bridge stream, then synchronize that stream exactly once at the
         * response/planning boundary.  Synchronizing `handle.stream` here is a
         * performance bug because later state-publication work may share it.
         */
        virtual bool materializeDeviceSpeculativeOutcomesForHostPlan(
            const DeviceSpeculativeOutcomeHandle &handle,
            DeviceSpeculativeVerifyBatchOutcome *outcomes)
        {
            return copyDeviceSpeculativeOutcomesToHost(handle, outcomes);
        }

        /**
         * @brief Materialize resident outcomes only for host-visible response data.
         *
         * Device-resident publication must already have consumed @p handle
         * before the all-position fast path calls this method.  The method may
         * copy compact output tokens and metadata so `decodeStep()` can return
         * tokens and update host diagnostics, but it must not be required for GPU
         * live-state mutation.
         */
        virtual bool materializeDeviceSpeculativeOutcomesForHostResponse(
            const DeviceSpeculativeOutcomeHandle &handle,
            DeviceSpeculativeVerifyBatchOutcome *outcomes)
        {
            return materializeDeviceSpeculativeOutcomesForHostPlan(
                handle,
                outcomes);
        }

        /**
         * @brief Enable GPU-side decode sampling mode
         *
         * When enabled, forward() may skip gathering logits to host for decode calls.
         * Caller should use sampleGreedyOnDevice() instead of logits().
         * Default: no-op (not all runners support this).
         */
        virtual void setSkipLogitsGatherDecode(bool) {}

        /**
         * @brief Skip logits gather after prefill (seq_len > 1)
         *
         * In the standard generation flow, prefill logits are never consumed —
         * the first generated token comes from a decode step. Skipping the
         * D2H logits gather for prefill eliminates massive PCIe traffic
         * (e.g. 346 MB for 596 tokens × 152064 vocab across 2 devices).
         * Default: no-op (not all runners support this).
         */
        virtual void setSkipLogitsGatherPrefill(bool) {}

        /**
         * @brief Suppress GPU stage timeline output
         *
         * When enabled, the GPU stage timeline summary is not printed after
         * each forward pass. Used by BenchmarkRunner to exclude warmup runs
         * from overhead reporting.
         * Default: no-op (not all runners support this).
         */
        virtual void setSuppressTimeline(bool) {}

        /**
         * @brief Set prefill accumulation mode for benchmark
         *
         * When enabled, prefill GPU stage timelines are accumulated across
         * iterations instead of being printed immediately. Used by
         * BenchmarkRunner to avoid per-iteration prefill table spam.
         * Default: no-op (not all runners support this).
         */
        virtual void setAccumulatePrefill(bool) {}

        /**
         * @brief Flush accumulated GPU stage timeline data
         *
         * Prints accumulated decode stage timing summary and resets.
         * Called after decode phase completes (e.g., by BenchmarkRunner).
         * Default: no-op (not all runners support this).
         */
        virtual void flushStageTimeline() {}

        /**
         * @brief Get current position in cache
         */
        virtual int get_position() const = 0;

        /**
         * @brief Get execution path being used
         */
        virtual ExecutionPath executionPath() const = 0;

        /**
         * @brief Get architecture name (e.g., "qwen2")
         */
        virtual const char *architecture() const = 0;

        // =====================================================================
        // Snapshot Capture API (for E2E parity testing)
        // =====================================================================
        // These methods have default no-op implementations for builds without
        // snapshot support. Override in Pipeline (with ENABLE_PIPELINE_SNAPSHOTS)
        // or DeviceGraphOrchestrator (always available) for actual snapshot capture.

        /**
         * @brief Enable snapshot capture of intermediate activations
         *
         * When enabled, each forward pass will capture intermediate tensor
         * values at instrumented stages for comparison against ground truth.
         *
         * @param output_dir Optional directory to save snapshots (implementation-specific)
         */
        virtual void enableSnapshotCapture(const std::string &output_dir = "")
        {
            (void)output_dir; // No-op by default
        }

        /**
         * @brief Disable snapshot capture and clear stored snapshots
         */
        virtual void disableSnapshotCapture() {}

        /**
         * @brief Clear stored snapshots but keep capture enabled
         */
        virtual void clearSnapshots() {}

        /**
         * @brief Retrieve a captured snapshot by key
         *
         * @param key Snapshot identifier (e.g., "layer0_Q_PROJECTION", "EMBEDDING")
         * @param out_size Output parameter for snapshot size in bytes
         * @return Pointer to snapshot data (FP32), or nullptr if key doesn't exist
         */
        virtual const float *getSnapshot(const std::string &key, size_t &out_size) const
        {
            (void)key;
            out_size = 0;
            return nullptr; // No snapshot support by default
        }

        /**
         * @brief Retrieve a captured snapshot with 2D shape metadata
         *
         * Returns the snapshot data along with the rows/cols that the stage
         * reported via getDumpInfo() at capture time. This allows callers to
         * understand the 2D layout without model-specific inference logic.
         *
         * @param key Snapshot identifier (e.g., "layer0_Q_PROJECTION")
         * @return SnapshotInfo with data pointer and shape, or empty if not found
         */
        virtual SnapshotInfo getSnapshotWithShape(const std::string &key) const
        {
            (void)key;
            return {};
        }

        /**
         * @brief Get list of all captured snapshot keys
         *
         * @return Vector of snapshot identifiers
         */
        virtual std::vector<std::string> getSnapshotKeys() const
        {
            return {}; // No snapshots by default
        }

        // =====================================================================
        // Hidden State API (for Pipeline Parallelism)
        // =====================================================================

        /**
         * @brief Get final hidden state from last forward pass
         *
         * Returns the hidden state tensor after all transformer layers have
         * executed. This is used for Pipeline Parallelism to transfer
         * activations between stages.
         *
         * @return Pointer to hidden state tensor [seq_len, d_model], or nullptr
         */
        virtual TensorBase *getHiddenState() { return nullptr; }
        virtual const TensorBase *getHiddenState() const { return nullptr; }

        /**
         * @brief Set initial hidden state for forward pass
         *
         * For PP stages that don't have embedding (middle/final stages),
         * this sets the hidden state that would normally come from embedding.
         * The forward pass will skip embedding and use this tensor directly.
         *
         * @param hidden_state Tensor containing hidden state [seq_len, d_model]
         */
        virtual void setHiddenState(TensorBase *hidden_state) { (void)hidden_state; }

        /**
         * @brief Check if this runner has hidden state set for next forward
         */
        virtual bool hasHiddenStateInput() const { return false; }

        /**
         * @brief Clear hidden state input (reset to normal embedding mode)
         */
        virtual void clearHiddenStateInput() {}

        // =====================================================================
        // Device & Logits Local API
        // =====================================================================
        // These methods expose device identity and column-parallel logits state
        // so that RankOrchestrator can coordinate logits gathering and
        // GPU-side sampling without downcasting to a concrete runner type.

        /**
         * @brief Get the primary device this runner executes on
         *
         * @return DeviceId of the primary compute device (CPU by default)
         */
        virtual DeviceId primaryDeviceId() const { return DeviceId::cpu(); }

        /**
         * @brief Check if this runner has column-parallel local logits
         *
         * True when the LM head is column-parallel and logits_local is allocated.
         *
         * @return true if getLogitsLocalInfo() will return valid info
         */
        virtual bool hasLogitsLocal() const { return false; }

        /**
         * @brief Get local logits info for column-parallel gathering
         *
         * Returns GPU pointer, device, shape, and tensor pointer for the
         * per-device logits shard. Used by RankOrchestrator for
         * AllGather of column-parallel LM head output.
         *
         * @return LogitsLocalInfo (empty by default)
         */
        virtual LogitsLocalInfo getLogitsLocalInfo() const { return {}; }

        /**
         * @brief Get local logits info for a sampling consumer.
         *
         * GPU decode graph replay can publish a one-shot producer stream for
         * the logits row.  Sampling must consume that stream so the argmax or
         * stochastic sampler is ordered after the graph replay that wrote the
         * row.  Plain gather/snapshot paths should continue to use
         * getLogitsLocalInfo(), which is a non-consuming view.
         *
         * @return LogitsLocalInfo with the correct consumer stream, or empty.
         */
        virtual LogitsLocalInfo consumeLogitsLocalInfoForSampling()
        {
            return getLogitsLocalInfo();
        }

        /**
         * @brief Check if this runner has column-parallel local MTP logits
         *
         * True when the MTP sidecar LM head writes a local vocabulary shard
         * instead of a full replicated logits row.
         */
        virtual bool hasMTPLogitsLocal() const { return false; }

        /**
         * @brief Get local MTP logits info for column-parallel gathering
         *
         * Uses the same LogitsLocalInfo contract as the main LM head so TP
         * orchestration can gather sidecar logits without knowing runner internals.
         */
        virtual LogitsLocalInfo getMTPLogitsLocalInfo() const { return {}; }

        /**
         * @brief Get local MTP logits info for a sampling consumer.
         *
         * MTP sidecar graphs can publish their logits on a sidecar capture or
         * replay stream.  The sampler must consume that producer stream, rather
         * than using a generic device stream, so sharded TP sampling observes
         * the logits written by the just-completed sidecar step.  Non-sampling
         * gather/snapshot paths should keep using getMTPLogitsLocalInfo().
         */
        virtual LogitsLocalInfo consumeMTPLogitsLocalInfoForSampling()
        {
            return getMTPLogitsLocalInfo();
        }

        // =====================================================================
        // Profiling API
        // =====================================================================

        /**
         * @brief Get executor statistics for profiling
         *
         * @return Pointer to GraphExecutorStats, or nullptr if not available
         */
        virtual const GraphExecutorStats *executorStats() const { return nullptr; }

        /**
         * @brief Reset executor statistics
         */
        virtual void resetExecutorStats() {}

        // =====================================================================
        // Orchestration API (for heterogeneous device placement)
        // =====================================================================

        /**
         * @brief Check if this runner has a PlacementPlan configured
         *
         * @return true if a PlacementPlan was provided during creation
         */
        virtual bool hasPlacementPlan() const { return false; }

        /**
         * @brief Get the PlacementPlan this runner is using
         *
         * @return Reference to the PlacementPlan
         * @throws std::runtime_error if no plan is configured (check hasPlacementPlan first)
         */
        virtual const PlacementPlan &getPlacementPlan() const
        {
            throw std::runtime_error("No PlacementPlan configured for this runner");
        }

        /**
         * @brief Domain-local MoE placement epoch used by prefix and graph caches.
         *
         * Non-MoE runners return 0. MoE runners increment this when expert
         * ownership, masks, replicas, or runtime-table placement changes.
         */
        virtual uint64_t moePlacementEpoch() const { return 0; }

        /**
         * @brief Enumerate MoE rebalance controllers owned by this runner.
         *
         * Single-device runners may own a primary controller plus routed-overlay
         * domain controllers. Composite runners return every local domain
         * controller so callers can avoid treating the first available device
         * controller as the multi-domain API.
         */
        virtual std::vector<MoERebalanceController *> moeRebalanceControllers() const { return {}; }

        /**
         * @brief Lookup a MoE rebalance controller by ExpertParallel domain id.
         */
        virtual MoERebalanceController *moeRebalanceControllerForDomain(
            const std::string &domain_id) const
        {
            (void)domain_id;
            return nullptr;
        }

        /**
         * @brief Participant index used for domain-scoped MoE rebalance actions.
         *
         * Single-device runners return 0. GlobalTP runners return rank-in-domain,
         * which is distinct from MPI local rank on multi-node or multi-domain runs.
         */
        virtual int moeRebalanceParticipantId() const { return 0; }

        /**
         * @brief Find the longest reusable prefix for a token sequence.
         *
         * Default runners do not support persistent prefix state yet. Concrete
         * implementations return a populated result only when the feature is
         * enabled and the active backend can import/export logical KV blocks.
         */
        virtual PrefixLookupResult lookupPrefix(const std::vector<int32_t> &tokens)
        {
            (void)tokens;
            return {};
        }

        virtual bool populatePrefix(const PrefixLookupResult &hit, int seq_idx = 0)
        {
            (void)hit;
            (void)seq_idx;
            return false;
        }

        virtual bool harvestPrefix(const std::vector<int32_t> &tokens, int prompt_token_count)
        {
            (void)tokens;
            (void)prompt_token_count;
            return false;
        }

        virtual bool restorePrefixTerminalState(const PrefixLookupResult &hit)
        {
            (void)hit;
            return false;
        }

        virtual PrefixStateSnapshot captureLivePrefixState(int seq_idx = 0) const
        {
            (void)seq_idx;
            return {};
        }

        virtual PrefixStateSnapshot captureLivePrefixCheckpoint(int seq_idx = 0) const
        {
            (void)seq_idx;
            return {};
        }

        virtual bool restoreLivePrefixState(const PrefixStateSnapshot &snapshot, int seq_idx = 0)
        {
            (void)snapshot;
            (void)seq_idx;
            return false;
        }

        virtual bool truncateLivePrefixState(int cached_tokens, int seq_idx = 0)
        {
            (void)cached_tokens;
            (void)seq_idx;
            return false;
        }

        /**
         * @brief Read-only runtime state probe for prefix-cache/MTP development.
         *
         * This is diagnostic state only: callers must not mutate runner-owned
         * buffers through the returned value.
         */
        virtual PrefixRuntimeStateSnapshot prefixStateProbe() const { return {}; }
    };

} // namespace llaminar2
