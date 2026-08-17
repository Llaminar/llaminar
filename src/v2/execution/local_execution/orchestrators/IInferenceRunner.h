/**
 * @file IInferenceRunner.h
 * @brief Interface for inference execution
 * @author David Sanftenberg
 * @date December 2025
 *
 * Interface implemented by DeviceGraphOrchestrator for inference execution.
 */

#pragma once

#include <algorithm>
#include <array>
#include <optional>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <stdexcept>

#include "../../../backends/DeviceId.h"
#include "../../moe/DeviceMoERebalanceABI.h"
#include "../../moe/MoEOverlayAuthorityExecution.h"
#include "../../mtp/MTPRejectionSampler.h"
#include "../../mtp/MTPVerifierOutcomeGraph.h"
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
    class MoEOverlayInferenceTransactionCoordinator;

    /**
     * @brief Frozen setup contract for the executable serving graph family.
     *
     * The orchestration memory plan is the authority for every physical
     * prefill shape admitted by a distributed ExpertOverlay cell.  Passing
     * that exact list into graph materialization prevents a device runner from
     * independently re-deriving capacity from local environment state.  The
     * padding token participates in native graph identity because it is an
     * immediate parameter of the captured chunk-materialization kernel.
     */
    struct ServingGraphFamilyMaterializationPlan
    {
        std::vector<int> prefill_bucket_rows; ///< Complete admitted physical bucket ladder.
        int prefill_pad_token_id = 0;         ///< Token written to inactive rows in every bucket.

        /** @brief True when the setup contract names at least one physical graph. */
        bool valid() const noexcept
        {
            return !prefill_bucket_rows.empty() &&
                   std::all_of(
                       prefill_bucket_rows.begin(),
                       prefill_bucket_rows.end(),
                       [](int rows)
                       { return rows > 0; });
        }
    };

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
        size_t vocab_offset = 0;        ///< Global token id represented by local column 0
        TensorBase *tensor = nullptr;   ///< Tensor pointer for CPU fallback (data())
        void *stream = nullptr;         ///< Explicit GPU stream (must match forward pass stream)

        // Device-resident scratch for the multi-block argmax reduction (owned by
        // the runner's BufferArena). Supplied to IBackend::argmaxF32() so the
        // CUDA two-pass reduction never has to allocate on the hot path. Null /
        // zero capacity means GPU-side argmax is unavailable for this runner.
        void *argmax_partial_vals = nullptr; ///< FP32 scratch [argmax_partial_capacity]
        void *argmax_partial_idxs = nullptr; ///< INT32 scratch [argmax_partial_capacity]
        int argmax_partial_capacity = 0;     ///< Number of entries in the scratch buffers
        size_t row_stride = 0;               ///< Physical row stride in floats; 0 means vocab_local

        /// True if this info is valid (has a tensor)
        explicit operator bool() const { return tensor != nullptr; }
    };

    enum class DeviceLogitsSource : uint8_t
    {
        Main,             ///< Ordinary one-row main-model terminal logits.
        MTP,              ///< NextN/MTP sidecar terminal logits.
        AllPosition,      ///< Speculative verifier target rows.
        MainRequestBatch, ///< Live grouped main-condition rows, one per request.
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
     * @brief Shared device-owned state for one request-scoped MTP transaction.
     *
     * GPU MTP shifted KV caches already own their live sequence metadata in
     * stable device allocations.  This object does not duplicate that state;
     * it names the canonical per-depth cached-token rows and carries the latest
     * stream/event ordering edge for mutations to those rows.  The object is
     * created for one runner request session and survives sidecar appends,
     * verifier publication, correction commits, and prefix restore.  Only a
     * request reset or cache replacement retires it.
     *
     * The state is shared because LocalTP keeps one outcome handle per child.
     * Publication may advance a child's fence after the rank has copied the
     * handle, and every copy must observe that newer fence without rebuilding
     * an ambient mailbox or retargeting a live-state epoch.
     */
    struct DeviceResidentMTPTransactionState
    {
        DeviceId device = DeviceId::invalid();
        int request_count = 0;
        std::vector<const int *> shifted_cached_tokens_device_by_depth;
        void *producer_stream = nullptr;
        std::shared_ptr<void> ready_event;
        uint64_t session_epoch = 0;
        uint64_t mutation_generation = 0;

        /**
         * @brief Return whether the transaction owns a canonical count row for a depth.
         */
        bool coversDepth(int depth) const
        {
            return depth >= 0 &&
                   depth < static_cast<int>(
                               shifted_cached_tokens_device_by_depth.size()) &&
                   shifted_cached_tokens_device_by_depth[
                       static_cast<size_t>(depth)] != nullptr;
        }

        /**
         * @brief Return the first request entry for one shifted-cache depth.
         */
        const int *cachedTokensForDepth(int depth) const
        {
            return coversDepth(depth)
                       ? shifted_cached_tokens_device_by_depth[
                             static_cast<size_t>(depth)]
                       : nullptr;
        }

        /**
         * @brief Return whether the transaction has complete ownership and ordering data.
         */
        bool valid() const
        {
            return device.is_gpu() &&
                   request_count > 0 &&
                   !shifted_cached_tokens_device_by_depth.empty() &&
                   std::all_of(
                       shifted_cached_tokens_device_by_depth.begin(),
                       shifted_cached_tokens_device_by_depth.end(),
                       [](const int *ptr) { return ptr != nullptr; }) &&
                   producer_stream != nullptr &&
                   ready_event != nullptr &&
                   mutation_generation > 0;
        }
    };

    /**
     * @brief Copyable child-local lease for a device-resident MTP transaction.
     *
     * A lease travels with compact verifier outcomes and resident logical-state
     * handles.  Consumers validate pointer identity against their owning
     * runner, then wait on the current fence stored in the shared state.  This
     * makes ownership explicit across RankOrchestrator fan-out while allowing
     * the transaction fence to advance after each canonical cache mutation.
     */
    struct DeviceResidentMTPTransactionLease
    {
        std::shared_ptr<DeviceResidentMTPTransactionState> state;

        bool valid() const
        {
            return state && state->valid();
        }

        bool coversRequest(int request_index) const
        {
            return valid() &&
                   request_index >= 0 &&
                   request_index < state->request_count;
        }
    };

    /**
     * @brief Device-resident compact outcome buffers for stochastic MTP.
     *
     * This is the first-class handoff object for the vLLM-style path where the
     * verifier summary remains on GPU.  The pointed-to buffers are owned by the
     * runner and are valid until the runner stages another stochastic outcome
     * request.  Device-resident publication consumes these buffers directly and
     * avoids every per-transaction D2H boundary. Production generation consumes
     * the handle only through device-resident publication. Focused diagnostics
     * may inspect it through copyDeviceSpeculativeOutcomesToHostForDiagnostics().
     */
    struct DeviceSpeculativeOutcomeHandle
    {
        const int32_t *output_tokens_device = nullptr;
        /**
         * Compact transaction metadata owned by the producer graph.
         *
         * Publication may terminally invalidate this row when response-ledger
         * commit fails.  Keeping the pointer mutable makes that single-device
         * authority explicit and prevents a later host bridge from observing a
         * stale successful transaction after publication has failed.
         */
        int *meta_device = nullptr;
        int request_count = 0;
        /**
         * @brief Number of real verifier rows in each logical request.
         *
         * This value is produced beside the compact outcome and is therefore
         * the only authority publication may use for commit-policy bounds.
         * Dynamic-depth scalar MTP can place this logical row count inside a
         * larger captured bucket; consumers must never infer it from graph
         * geometry or re-declare it in a later handoff.
         */
        int logical_verifier_rows_per_request = 0;
        /**
         * @brief Immutable row stride owned by the captured verifier graph.
         *
         * For scalar GPU MTP this may be the next power-of-two bucket above
         * @ref logical_verifier_rows_per_request. Request-batched and CPU
         * producers use exact geometry. The producer seals both values into
         * this handle so publication can address physical verifier state while
         * committing only the logical prefix.
         */
        int physical_verifier_rows_per_request = 0;
        int output_token_stride = sampling_math::kSpeculativeBatchMaxOutputTokens;
        int meta_stride = sampling_math::kSpeculativeBatchMetaCount;
        DeviceId device;
        void *stream = nullptr;
        /**
         * @brief Event recorded after compact verifier outcome rows are ready.
         *
         * State publication may enqueue more work on @ref stream after this
         * handle is returned.  Host response materialization should wait on
         * this event before copying compact response rows, rather than
         * synchronizing @ref stream directly, so the served-token bridge does
         * not accidentally drain later live-state publication work.
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
        /**
         * @brief Child-local ownership lease for shifted MTP KV state.
         *
         * Every participant retains the lease produced beside its own compact
         * token/meta buffers. Initial/suffix shifted-row commits therefore
         * consume the same cache owner that produced the child verifier outcome.
         */
        DeviceResidentMTPTransactionLease mtp_transaction;
        /**
         * @brief Whether the compact row participates in the resident response ledger.
         *
         * Every production GPU grouped reducer sets this bit only after captured
         * verifier preparation has consumed the request-admission event and
         * published a device-owned transaction budget. Accepted-state
         * publication must then use the fused response-commit/publication kernel
         * for both greedy and stochastic sampling.
         */
        bool device_generation_controller_owned = false;
        /**
         * @brief True when this child produced a complete mirrored LocalTP outcome.
         *
         * Every child in a mirrored domain must report this value. It proves
         * that the child ran the participant-local terminal reducer and owns a
         * ready compact outcome; no rank outcome collective is permitted.
         */
        bool mirrored_local_tp_locally_complete = false;

        bool valid() const
        {
            return output_tokens_device != nullptr &&
                   meta_device != nullptr &&
                   request_count > 0 &&
                   logical_verifier_rows_per_request > 0 &&
                   physical_verifier_rows_per_request >=
                       logical_verifier_rows_per_request &&
                   (request_count == 1 ||
                    physical_verifier_rows_per_request ==
                        logical_verifier_rows_per_request) &&
                   output_token_stride > 0 &&
                   meta_stride >= sampling_math::kSpeculativeBatchMetaCount &&
                   stream != nullptr &&
                   response_ready_event != nullptr;
        }
    };

    /**
     * @brief Immutable request used to admit a device-owned generation ledger.
     *
     * The leading-row disposition is part of response correctness, not optional
     * scheduling metadata. A controller reopened after a rejection must consume
     * the already-emitted correction as verifier row zero without returning it
     * twice. Keeping geometry, response budget, and row ownership in one value
     * prevents rank and backend layers from dropping that state independently.
     */
    struct DeviceGenerationAdmissionRequest
    {
        int request_count = 0; ///< Number of independent device controller rows.
        int max_new_tokens = 0; ///< New response-token capacity for each row.
        sampling_math::DeviceGenerationLeadingRowDisposition
            initial_leading_row_disposition =
                sampling_math::DeviceGenerationLeadingRowDisposition::
                    PendingResponse; ///< Uniform row-zero ownership at admission.

        /** @return true when every field describes a legal controller admission. */
        [[nodiscard]] bool valid() const noexcept
        {
            return request_count > 0 && max_new_tokens > 0 &&
                   sampling_math::
                       valid_device_generation_leading_row_disposition(
                           initial_leading_row_disposition);
        }
    };

    /**
     * @brief Host-visible terminal record for one device-owned generation row.
     *
     * This record is created only after the request's final controller
     * publication has been consumed through an explicit GPU event edge.  It is
     * deliberately a terminal result rather than a live-state mirror: none of
     * these fields may feed a later graph replay or accepted-state publication.
     */
    struct DeviceGenerationTerminalRequestResult
    {
        std::vector<int32_t> tokens; ///< Exact response tokens emitted by the device ledger.
        int remaining_token_count = 0; ///< Unused response budget when a stop token ended generation.
        bool model_stopped = false; ///< True when generation ended on the request stop policy.
        sampling_math::DeviceGenerationLeadingRowDisposition
            next_leading_row_disposition =
                sampling_math::DeviceGenerationLeadingRowDisposition::
                    PendingResponse; ///< Ownership of the retained continuation row.
        int transaction_count = 0; ///< Number of committed speculative transactions.
        int accepted_speculative_token_count = 0; ///< Accepted draft-token total.
        int rejected_transaction_count = 0; ///< Transactions that emitted a rejection correction.
        int consumed_verifier_row_count = 0; ///< Total verifier rows consumed by committed transactions.
        int published_state_commit_count = 0; ///< Total main-graph state rows committed by the device.
        int attempted_draft_token_count = 0; ///< Sum of device-selected draft widths.
        int verifier_token_count = 0; ///< Sum of logical verifier widths, including condition rows.
        int last_transaction_draft_depth = 0; ///< Selected width of the final committed transaction.
        int last_transaction_emitted_token_count = 0; ///< Response width of the final transaction.
        int final_draft_depth = 0; ///< Device selector at the terminal boundary.
        int depth_evaluated_window_count = 0; ///< Device-owned dynamic-policy windows evaluated.
        int depth_update_count = 0; ///< Applied dynamic selector transitions.
        int depth_promotion_count = 0; ///< Applied one-step promotions.
        int depth_demotion_count = 0; ///< Applied one-step demotions.

        bool operator==(
            const DeviceGenerationTerminalRequestResult &) const = default;
    };

    /**
     * @brief Complete terminal response surfaced from a resident GPU generation.
     *
     * A successful result proves that every request controller was healthy,
     * complete, and internally consistent.  The response/control D2H copies are
     * queued together and observed through one terminal stream synchronization;
     * no per-transaction bridge is part of this contract.
     */
    struct DeviceGenerationTerminalResult
    {
        DeviceId device = DeviceId::invalid(); ///< Device that owned the authoritative ledger.
        std::vector<DeviceGenerationTerminalRequestResult> requests;

        bool valid() const
        {
            return device.is_gpu() && !requests.empty();
        }
    };

    /**
     * @brief Sampling topology embedded in a native device-generation parent.
     *
     * Greedy grouped verification reduces compact outcomes inside the retained
     * all-position forward graph. Stochastic grouped verification instead owns
     * separate target-distribution and serial-rejection child graphs. The two
     * parent bodies are therefore different executables even when their request
     * and verifier geometry match. Carrying this closed mode through the public
     * runner contract prevents either topology from being guessed from ambient
     * sampling parameters or reused under the other's cache identity.
     */
    enum class DeviceGenerationSamplingMode : uint8_t
    {
        Greedy = 0,
        Stochastic = 1,
    };

    /**
     * @brief Return the stable diagnostic name for a device-generation mode.
     */
    constexpr const char *deviceGenerationSamplingModeName(
        DeviceGenerationSamplingMode mode) noexcept
    {
        switch (mode)
        {
        case DeviceGenerationSamplingMode::Greedy:
            return "greedy";
        case DeviceGenerationSamplingMode::Stochastic:
            return "stochastic";
        }
        return "invalid";
    }

    /**
     * @brief Validate a possibly deserialized device-generation mode.
     */
    constexpr bool isValidDeviceGenerationSamplingMode(
        DeviceGenerationSamplingMode mode) noexcept
    {
        return mode == DeviceGenerationSamplingMode::Greedy ||
               mode == DeviceGenerationSamplingMode::Stochastic;
    }

    /**
     * @brief Control topology required by one complete MTP generation loop.
     *
     * Fixed-depth generation needs one device-controlled WHILE body. Dynamic
     * generation additionally needs a device-selected branch for every legal
     * draft depth. Keeping this distinction typed prevents callers from
     * assuming that support for a fixed conditional body also proves support
     * for a SWITCH-in-WHILE graph.
     */
    enum class DeviceGenerationLoopTopology : uint8_t
    {
        FixedDepth = 0, ///< One immutable transaction body repeated to completion.
        DynamicDepth,   ///< Device-selected transaction body repeated to completion.
    };

    /**
     * @brief Backend execution policy for the complete MTP generation loop.
     *
     * `NativeConditionalGraph` is the target architecture: one asynchronous
     * parent graph owns every transaction. HIP currently exposes neither graph
     * conditional nodes nor device-side graph launch, so ROCm deliberately uses
     * `HostScheduledCapturedTransactions`: each expensive transaction remains a
     * captured GPU graph, while the host advances only the outer transaction
     * loop from a narrow generation-tagged dispatch ticket. Compact outcomes,
     * response ledgers, caches, samplers, and dynamic-depth state remain device
     * authoritative. This is an explicit backend policy, not a retry after
     * native graph construction fails.
     *
     * `Unsupported` is fatal. It distinguishes a known backend limitation from
     * a missing capture owner or an unexpectedly incomplete native capability.
     */
    enum class DeviceGenerationExecutionPolicy : uint8_t
    {
        NativeConditionalGraph = 0,
        HostScheduledCapturedTransactions,
        Unsupported,
    };

    /**
     * @brief Backend policy for one committed device MoE maintenance edge.
     *
     * CUDA owns the complete `publish -> IF(due) maintenance -> acknowledge`
     * graph on device. HIP lacks conditional graph nodes, so homogeneous ROCm
     * domains use a rank-validated ticket to choose between already captured
     * maintenance and acknowledgement graphs. `Inactive` means this runner has
     * no homogeneous device-owned rebalance controller; `Unsupported` is a
     * fatal incomplete implementation, never permission to select another path.
     */
    enum class DeviceMoERebalanceMaintenanceExecutionPolicy : uint8_t
    {
        Inactive = 0,
        NativeConditionalGraph,
        HostScheduledCapturedMaintenance,
        Unsupported,
    };

    /**
     * @brief Conservative HIP ticket-observation cadence from immutable setup.
     *
     * This is not a host mirror of the device controller. The two round
     * intervals are immutable graph configuration and the per-boundary bound
     * is the largest commit that the configured serial/MTP transaction can
     * make. Together they prove how many complete decode transactions can run
     * through a captured device-only publish/ack graph before maintenance could
     * possibly become due. The eventual ticket remains the sole live decision.
     */
    struct DeviceMoERebalanceHostedObservationSchedule
    {
        uint32_t initial_round_interval = 0;
        uint32_t recurring_round_interval = 0;
        uint32_t maximum_committed_rounds_per_boundary = 0;

        /** @brief Return true when every cadence divisor is usable. */
        constexpr bool valid() const noexcept
        {
            return initial_round_interval > 0u &&
                   recurring_round_interval > 0u &&
                   maximum_committed_rounds_per_boundary > 0u;
        }

        /**
         * @brief Boundaries required before @p remaining_rounds can reach zero.
         *
         * The ceiling division deliberately permits an early observation when
         * MTP accepts fewer rows than its configured maximum. It can never
         * schedule an observation after a due edge.
         */
        constexpr uint32_t boundariesUntilPotentiallyDue(
            uint32_t remaining_rounds) const noexcept
        {
            return valid() && remaining_rounds > 0u
                       ? 1u +
                             (remaining_rounds - 1u) /
                                 maximum_committed_rounds_per_boundary
                       : 0u;
        }

        friend constexpr bool operator==(
            const DeviceMoERebalanceHostedObservationSchedule &,
            const DeviceMoERebalanceHostedObservationSchedule &) = default;
    };

    /** @brief Stable diagnostic name for a device MoE scheduler policy. */
    constexpr const char *deviceMoERebalanceMaintenanceExecutionPolicyName(
        DeviceMoERebalanceMaintenanceExecutionPolicy policy) noexcept
    {
        switch (policy)
        {
        case DeviceMoERebalanceMaintenanceExecutionPolicy::Inactive:
            return "inactive";
        case DeviceMoERebalanceMaintenanceExecutionPolicy::NativeConditionalGraph:
            return "native_conditional_graph";
        case DeviceMoERebalanceMaintenanceExecutionPolicy::HostScheduledCapturedMaintenance:
            return "host_scheduled_captured_maintenance";
        case DeviceMoERebalanceMaintenanceExecutionPolicy::Unsupported:
            return "unsupported";
        }
        return "invalid";
    }

    /**
     * @brief Return the stable diagnostic name for an MTP loop policy.
     */
    constexpr const char *deviceGenerationExecutionPolicyName(
        DeviceGenerationExecutionPolicy policy) noexcept
    {
        switch (policy)
        {
        case DeviceGenerationExecutionPolicy::NativeConditionalGraph:
            return "native_conditional_graph";
        case DeviceGenerationExecutionPolicy::HostScheduledCapturedTransactions:
            return "host_scheduled_captured_transactions";
        case DeviceGenerationExecutionPolicy::Unsupported:
            return "unsupported";
        }
        return "invalid";
    }

    /**
     * @brief Static graph shape for publication from a device outcome row.
     *
     * Dynamic positions, cache lengths, accepted counts, request cardinality,
     * logical verifier width, physical captured stride, and shifted-cache
     * ownership all come from the producer-owned outcome handle. The caller
     * supplies only commit policy; there is intentionally no duplicate geometry,
     * host position, or alternate cache-count payload in this request.
     */
    struct DeviceSpeculativePublicationRequest
    {
        DeviceSpeculativeOutcomeHandle outcome;
        /**
         * @brief Maximum verifier prefix allowed to become serial-visible state.
         *
         * This is deliberately separate from the outcome's logical and physical
         * verifier widths. An all-accepted terminal row may be valid speculative
         * evidence while still being one row beyond the caller-visible serial
         * state.
         */
        int max_state_commit_rows = -1;
        bool publish_mtp_shifted_kv = true;
        /**
         * @brief Immutable request policy validated at the publication edge.
         *
         * Accepted-state publication is the sole owner of generated-token
         * histogram advancement for both greedy and stochastic outcomes.  The
         * evolving pending-condition predicate is deliberately absent from
         * this host request: the publication kernel derives and stores it from
         * final device metadata on the same stream as the state commit.
         */
        MTPRequestPenaltyPolicy penalty_policy{};

        [[nodiscard]] int requestCount() const noexcept
        {
            return outcome.request_count;
        }

        [[nodiscard]] int logicalVerifierRowsPerRequest() const noexcept
        {
            return outcome.logical_verifier_rows_per_request;
        }

        [[nodiscard]] int physicalVerifierRowsPerRequest() const noexcept
        {
            return outcome.physical_verifier_rows_per_request;
        }

        bool valid() const
        {
            return outcome.valid() &&
                   outcome.mtp_transaction.valid() &&
                   max_state_commit_rows >= 0 &&
                   max_state_commit_rows <=
                       logicalVerifierRowsPerRequest() &&
                   (!penalty_policy.enabled() || requestCount() == 1);
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
     * The pointed-to buffers are owned by the runner's persistent arena. They are
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
        /// Monotonic identity of the exact publication protected by ready_event.
        uint64_t publication_generation = 0;
        /// Child-local shifted-cache transaction consumed by correction sidecars.
        DeviceResidentMTPTransactionLease mtp_transaction;

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
                   ready_event != nullptr &&
                   publication_generation > 0;
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
         * field plus the publication generation prevents a later publication,
         * workspace rebind, or reset from accidentally reusing an old sidecar
         * replay after the shared event object has been recorded again.
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
                   live_state_epoch == other.live_state_epoch &&
                   publication_generation == other.publication_generation &&
                   mtp_transaction.state == other.mtp_transaction.state;
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
     * @brief Device-owned sampled-token mailbox slot.
     *
     * LocalTP MTP keeps target and draft samples in runner-owned device
     * mailboxes so the next graph can consume the token without a host round
     * trip. The generic handle is retained for main-target coordination when
     * the declarative final-head policy remains column parallel. Mirrored MTP
     * draft heads deliberately expose no rank-broadcast handle: every graph
     * participant samples and publishes its own local slot. This handle exposes
     * only the token pointer, exact producer stream, and ownership metadata
     * needed to reject stale or cross-device use.
     */
    struct DeviceStochasticSampleSlotHandle
    {
        int32_t *token_device = nullptr;
        int slot = -1;
        DeviceId device = DeviceId::invalid();
        void *stream = nullptr;

        bool valid() const
        {
            return token_device != nullptr &&
                   slot >= 0 &&
                   device.is_valid() &&
                   stream != nullptr;
        }
    };

    /// Main-target slots may still require coordination under a sharded head policy.
    using DeviceStochasticTargetSampleSlotHandle =
        DeviceStochasticSampleSlotHandle;

    /**
     * @brief Authority that supplies logical positions for seeded GPU MTP draws.
     *
     * Seeded speculative sampling is batch invariant only when every draw is
     * keyed by the logical output position that serial decode would use.  This
     * enum makes the owner of that position explicit; in particular, a GPU
     * request cannot accidentally carry both a host scalar and a device pointer.
     */
    enum class DeviceStochasticDrawPositionSource : uint8_t
    {
        /** The descriptor carries explicit threshold arrays; no position is read. */
        ExplicitThresholds,
        /** The descriptor carries inverse_sample_first_logical_position. */
        HostLogicalPosition,
        /** Read the request row from DeviceResidentLogicalSequenceStateHandle. */
        ResidentLogicalState,
        /** Read the pre-verifier device-to-device KV base-count snapshot. */
        VerifierBaseSnapshot,
    };

    /**
     * @brief One logical request inside a device-side stochastic MTP batch.
     *
     * The descriptor is intentionally value-owned: thresholds and stop tokens
     * are copied into fixed-size arrays before the runner call. That keeps the
     * request-batch handoff atomic and avoids dangling pointers when the caller
     * builds several requests before a GPU reducer consumes them.  Seeded
     * vLLM-style verification can set @ref derive_thresholds_from_seed to let
     * the GPU derive accept, residual, target-sample, and bonus-sample draws
     * from @ref inverse_sample_seed and the authority selected by
     * @ref draw_position_source.  Device position sources are read by kernels
     * on the explicit verifier stream; no host threshold array is populated.
     *
     * Seeded serial-equivalent stochastic verification sets
     * @ref serial_sample_equivalent instead. In that mode the verifier samples
     * each target row with either an explicit @ref sample_thresholds entry or a
     * seed-derived device-position draw, compares those sampled target tokens
     * against the materialized draft-token row, and summarizes with the same
     * compact metadata ABI as greedy MTP. This is the batch-invariant path: it
     * proves the grouped verifier produced the same tokens as serial stochastic
     * decode at the same logical positions.
     */
    struct DeviceStochasticBatchOutcomeRequest
    {
        int request_id = -1;          ///< Logical request id for diagnostics.
        int first_target_slot = -1;   ///< First verifier target row slot.
        int first_draft_slot = -1;    ///< First sampled draft-token slot.
        int row_count = 0;            ///< Number of speculative rows to compare.
        int32_t first_token = -1;     ///< First main-model token, if host-owned.
        bool first_token_from_device = false; ///< Read first token from exactly one device-owned source.
        /**
         * @brief Source slot while the first token is still owned by the target sampler.
         *
         * A device-owned first token names either this slot or @ref token_row_offset,
         * never both. Once verifier-input preparation copies a deferred sample into
         * entry zero of a materialized verifier row, ownership transfers to that row
         * and this field must return to `-1`.
         */
        int first_target_sample_slot = -1;
        /**
         * @brief Entry zero of the materialized `[first_token, draft...]` row.
         *
         * This offset is both the first-token source after materialization and the
         * row base consumed by serial-equivalent compact summaries. It is mutually
         * exclusive with @ref first_target_sample_slot when
         * @ref first_token_from_device is true.
         */
        int token_row_offset = -1;
        int token_row_stride = 0; ///< Prepared verifier-token row stride.
        int bonus_target_slot = -1;           ///< Bonus row slot, or -1.
        float bonus_threshold = 0.0f;         ///< Explicit bonus draw; ignored for seeded device-position draws.
        uint64_t inverse_sample_seed = 0;     ///< Shared seed for verifier draws and rejection inverse sampling.
        int inverse_sample_first_logical_position = 0;
        bool use_vllm_probability_rejection = false;
        bool derive_thresholds_from_seed = false;
        DeviceStochasticDrawPositionSource draw_position_source =
            DeviceStochasticDrawPositionSource::ExplicitThresholds;
        bool serial_sample_equivalent = false;
        bool use_device_draft_tokens = true; ///< Null host draft pointer when true.
        std::vector<int32_t> draft_tokens;
        std::vector<float> accept_thresholds;
        std::vector<float> residual_thresholds;
        std::vector<float> sample_thresholds;
        std::array<int32_t, sampling_math::kSpeculativeBatchMaxStopTokens> stop_tokens;
        int stop_token_count = 0;

        DeviceStochasticBatchOutcomeRequest()
            : draft_tokens(
                  static_cast<size_t>(sampling_math::kSpeculativeBatchMaxRows),
                  -1),
              accept_thresholds(
                  static_cast<size_t>(sampling_math::kSpeculativeBatchMaxRows),
                  0.0f),
              residual_thresholds(
                  static_cast<size_t>(sampling_math::kSpeculativeBatchMaxRows),
                  0.0f),
              sample_thresholds(
                  static_cast<size_t>(sampling_math::kSpeculativeBatchMaxRows),
                  0.0f)
        {
            stop_tokens.fill(-1);
        }

        /**
         * @brief Ensure optional host diagnostic rows can describe `count` rows.
         *
         * Seed-derived GPU production requests ordinarily leave these vectors
         * untouched because tokens and thresholds stay device-owned. Explicit
         * CPU/diagnostic callers use this helper before writing row values so
         * a configured speculative depth is not constrained by the default
         * certification depth.
         */
        bool ensureHostRowCapacity(int count)
        {
            if (count < 0)
                return false;
            const size_t required = static_cast<size_t>(count);
            if (draft_tokens.size() < required)
                draft_tokens.resize(required, -1);
            if (accept_thresholds.size() < required)
                accept_thresholds.resize(required, 0.0f);
            if (residual_thresholds.size() < required)
                residual_thresholds.resize(required, 0.0f);
            if (sample_thresholds.size() < required)
                sample_thresholds.resize(required, 0.0f);
            return true;
        }

        /** @brief Return whether every optional host row vector covers `count`. */
        bool hasHostRowCapacity(int count) const
        {
            if (count < 0)
                return false;
            const size_t required = static_cast<size_t>(count);
            return draft_tokens.size() >= required &&
                   accept_thresholds.size() >= required &&
                   residual_thresholds.size() >= required &&
                   sample_thresholds.size() >= required;
        }

        const int32_t *hostDraftTokensOrNull() const
        {
            return use_device_draft_tokens ? nullptr : draft_tokens.data();
        }
    };

    /**
     * @brief One row in a device-resident MTP verifier token batch.
     *
     * Request-batched GPU MTP verifier forwards consume a padded INT32 matrix
     * laid out as `[request_count, padded_seq_len]`.  Each descriptor tells the
     * runner how to compose one logical row on the graph replay stream:
     *
     * - entry 0 is a canonical device target-sample slot. The initial target
     *   sampler writes it directly and accepted-state publication refreshes it
     *   in the same captured kernel that publishes the logical-state mailbox;
     *   the host scalar is diagnostic metadata only and is never a GPU input;
     * - entries 1..N are copied from runner-owned draft sample slots;
     * - the returned matrix pointer is the only token source used by the verifier
     *   embedding graph.
     */
    struct DeviceMTPVerifierInputBatchRequest
    {
        int request_id = -1; ///< Logical request id for diagnostics.
        int32_t first_token = -1; ///< Optional host shadow for diagnostics only.
        bool first_token_from_device = false; ///< Must be true for every GPU verifier row.
        int first_target_sample_slot = -1; ///< Canonical target slot for row entry 0.
        int first_draft_slot = -1; ///< First device draft slot copied into row entry 1.
        int draft_token_count = 0; ///< Number of draft tokens copied after entry 0.
        int total_verifier_input_tokens = 0; ///< Valid row width before padding.
    };

    /**
     * @brief One logical request inside a resident greedy outcome batch.
     *
     * The verifier graph has already produced compact all-position logits and the
     * device-token matrix has already been materialized.  This descriptor binds
     * one request's compact logit rows to the matching verifier-token matrix row
     * so the backend can run argmax and the greedy acceptance reducer entirely on
     * device.
     */
    struct DeviceGreedyBatchOutcomeRequest
    {
        int request_id = -1; ///< Logical request id for diagnostics.
        int first_target_row = -1; ///< First compact all-position logit row.
        int verifier_token_count = 0; ///< Total verifier rows: drafts plus bonus.
        int token_row_offset = -1; ///< INT32 offset in the prepared token matrix.
        int token_row_stride = 0; ///< INT32 stride between prepared token rows.
        int32_t first_token = -1; ///< Host shadow for diagnostics/backend ABI.
        int leading_committed_output_count = 0; ///< Zero, or one pending correction row.
        std::array<int32_t, sampling_math::kSpeculativeBatchMaxStopTokens> stop_tokens;
        int stop_token_count = 0;

        DeviceGreedyBatchOutcomeRequest()
        {
            stop_tokens.fill(-1);
        }
    };

    /**
     * @brief Lightweight view of a captured snapshot with 2D shape metadata
     *
     * Returned by getSnapshotWithShape() to provide shape information
     * alongside the raw FP32 data pointer. The shape (rows, cols) comes
     * from the stage's getDumpInfo() at capture time, so stages own their
     * own dimension reporting. Production snapshot providers also attach a
     * type-erased lifetime token: code that retains SnapshotInfo across a
     * later graph callback, reset, or map replacement keeps the exact
     * captured tensor alive through that token.
     */
    struct SnapshotInfo
    {
        const float *data = nullptr; ///< Pointer to immutable FP32 snapshot data.
        size_t size = 0;             ///< Total element count (rows * cols)
        size_t rows = 0;             ///< Logical rows (e.g. seq_len)
        size_t cols = 0;             ///< Logical cols (e.g. hidden_dim, kv_dim, d_ff)

        /**
         * @brief Retains the publication backing data when the provider has ownership.
         *
         * SnapshotInfo remains a lightweight view, so this type-erased token
         * avoids exposing SnapshotCapture storage through every runner
         * interface. A null token is permitted for older mock/static providers
         * whose runner itself owns the data for the caller's whole operation.
         */
        std::shared_ptr<const void> lifetime_owner;

        explicit operator bool() const { return data != nullptr && size > 0; }
    };

    /**
     * @brief Explicit reset contract for request-owned inference state.
     *
     * Historically this boundary was named `clear_cache()`, which hid several
     * different lifetimes behind one word. The live sequence state is actually
     * owned by four independent families:
     * - main KV plus hybrid recurrent state (GDN/short-conv) owned by the
     *   model runner's live sequence;
     * - MTP sidecar KV, verifier/publication mailboxes, and speculative
     *   handoffs owned by the active MTP transaction;
     * - model-local request state such as MoE runtime placement tables and
     *   decode/rebalance histograms owned by the graph builder;
     * - logical positions, sequence lengths, and terminal-row metadata owned
     *   by the request boundary.
     *
     * A reset request names which owners cross the boundary and whether
     * replay-safe graph captures may remain alive. Implementations must fail
     * loudly when asked for a boundary they cannot represent; they must not
     * silently degrade into a broader or narrower reset.
     */
    struct InferenceStateResetRequest
    {
        /**
         * @brief Semantic boundary that initiated the reset.
         */
        enum class Boundary
        {
            Request,       ///< New prompt/session or benchmark iteration.
            PrefixRestore, ///< Prefix hit import replaces live state.
            HardReset,     ///< Replay/workspace teardown; no graph preservation.
        };

        Boundary boundary = Boundary::Request;
        bool reset_kv = true;               ///< Clear main and PP KV payloads.
        bool reset_gdn = true;              ///< Clear hybrid GDN/short-conv payloads.
        bool reset_mtp = true;              ///< Clear MTP sidecars and transaction state.
        bool reset_model_runtime = true;    ///< Clear graph-owned MoE/model request state.
        bool reset_logical_sequence = true; ///< Clear positions, lengths, terminal rows.
        bool preserve_replay_safe_graphs = true; ///< Keep proven replay-safe graph captures.
        const char *reason = "request-boundary";

        /**
         * @brief Build the standard new-request reset used by serving and tests.
         */
        static InferenceStateResetRequest requestBoundary(const char *why)
        {
            InferenceStateResetRequest request;
            request.boundary = Boundary::Request;
            request.reset_kv = true;
            request.reset_gdn = true;
            request.reset_mtp = true;
            request.reset_model_runtime = true;
            request.reset_logical_sequence = true;
            request.preserve_replay_safe_graphs = true;
            request.reason = why ? why : "request-boundary";
            return request;
        }

        /**
         * @brief Build the standard prefix-restore reset.
         *
         * Prefix restore is a replacement of live request state with a cached
         * prefix snapshot.  KV/GDN/MTP/logical sequence owners are always
         * cleared before importing the snapshot. Model-local runtime state is
         * restored to its immutable baseline only when the prefix entry has no
         * explicit model-runtime snapshot; otherwise the caller restores that
         * owner immediately after this reset. Both paths mutate contents behind
         * model-lifetime device addresses and publish one event, so replay-safe
         * graph captures remain valid.
         */
        static InferenceStateResetRequest prefixRestoreBoundary(
            const char *why,
            bool reset_model_runtime_owner)
        {
            InferenceStateResetRequest request;
            request.boundary = Boundary::PrefixRestore;
            request.reset_kv = true;
            request.reset_gdn = true;
            request.reset_mtp = true;
            request.reset_model_runtime = reset_model_runtime_owner;
            request.reset_logical_sequence = true;
            request.preserve_replay_safe_graphs = true;
            request.reason = why ? why : "prefix-restore";
            return request;
        }

        /**
         * @brief Return true when every live request-state owner crosses.
         */
        bool resetsAllLiveRequestOwners() const
        {
            return reset_kv && reset_gdn && reset_mtp && reset_logical_sequence;
        }
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
         * @brief Wait for the most recently submitted forward pass at a benchmark boundary.
         *
         * Production GPU inference deliberately returns after publishing its exact
         * terminal event so downstream device work can remain asynchronous. A host
         * wall-clock benchmark, however, must not stop its timer at graph submission.
         * GPU runners override this method and wait only for the durable event
         * that terminates the complete forward transaction. For MTP prefill,
         * that boundary includes shifted sidecar KV population rather than only
         * the earlier main-graph output. Implementations must not synchronize the
         * whole device or copy logits to host. CPU execution is already complete
         * when forward() returns.
         *
         * @return true when the last forward pass is complete and may be timed.
         */
        virtual bool waitForLastForwardCompletionForBenchmark()
        {
            return !primaryDeviceId().is_gpu();
        }

        /**
         * @brief Run a prompt/suffix prefill forward pass.
         *
         * Unlike generic forward(), this must keep prefill phase semantics even
         * when the request already has a nonzero cached position. Prefix-cache
         * partial hits use this for suffix prefill; MTP verifier continuations
         * should continue to use forward() so they can request decode-equivalent
         * short-continuation behavior explicitly.
         */
        virtual bool forwardPrefill(const int *tokens, int seq_len)
        {
            return forward(tokens, seq_len);
        }

        /**
         * @brief Capture and instantiate the complete serving graph family.
         *
         * This setup-only operation must not execute model arithmetic, mutate
         * request/KV state, publish sparse tickets, or advance a transaction.
         * GPU implementations retain the resulting native executables so the
         * first admitted request is an ordinary replay. Composite runners must
         * invoke every symmetric LocalTP participant concurrently.
         *
         * @param plan Frozen orchestration-owned physical graph inventory.
         * @return True only when every required executable is resident.
         */
        virtual bool materializeServingGraphFamilyWithoutLaunch(
            const ServingGraphFamilyMaterializationPlan &plan)
        {
            (void)plan;
            return false;
        }

        /**
         * @brief Install the request generation used by graph-native MoE sparse collectives.
         *
         * Distributed ExpertOverlay graphs are materialized independently on
         * their continuation and expert ranks. The orchestration layer supplies
         * one non-zero generation at each request boundary so graph capture or
         * cache lifetime can never become part of a sparse wire key. Runners
         * that do not implement graph-native ExpertOverlay return false.
         *
         * @param generation_id Monotonic root-authoritative request generation.
         * @return True only when the runner accepted the immutable generation.
         */
        virtual bool setMoEOverlayCollectiveRequestGeneration(
            uint64_t generation_id)
        {
            (void)generation_id;
            return false;
        }

        /**
         * @brief Bind the rank-wide heterogeneous ExpertOverlay ticket authority.
         *
         * A single-device continuation uses participant index zero. Composite
         * rank runners propagate the same shared coordinator to every local
         * graph with a distinct stable index. The coordinator publishes one
         * remote ticket only after authenticating symmetric graph geometry;
         * remote expert-only runners never receive this source-side binding.
         *
         * @param coordinator Setup-owned rank transaction coordinator.
         * @param continuation_participant_index Stable local graph index.
         * @return True only when this runner can publish at real graph edges.
         */
        virtual bool setMoEOverlayInferenceTransactionCoordinator(
            std::shared_ptr<MoEOverlayInferenceTransactionCoordinator>
                coordinator,
            int continuation_participant_index)
        {
            (void)coordinator;
            (void)continuation_participant_index;
            return false;
        }

        /**
         * @brief Run one CPU grouped-MTP verifier transaction from host token rows.
         *
         * Grouped verification is a semantic execution role rather than an
         * inference-shape heuristic. Implementations must therefore build the
         * graph with `ForwardExecutionRole::GroupedMTPVerifier`, including the
         * recurrent-state capture slots consumed by accepted-state publication.
         * The rectangular batch may contain rows of different logical lengths;
         * implementations own padding and must preserve each row's exact length.
         *
         * This entrypoint is intentionally host-resident and CPU-only. GPU
         * production paths must reject it and use the device-token entrypoints
         * below so token IDs never cross the host boundary during inference.
         *
         * @param token_batches Logical verifier token rows, one per request.
         * @return true when the grouped verifier forward succeeds.
         */
        virtual bool forwardGroupedMTPVerifierWithHostTokenIds(
            const std::vector<std::vector<int>> &token_batches)
        {
            (void)token_batches;
            return false;
        }

        /**
         * @brief Run one grouped MTP verifier pass from device-resident token IDs.
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
        virtual bool forwardGroupedMTPVerifierWithDeviceTokenIds(
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
         * @brief Advance one GPU main-model condition row from a resident mailbox.
         *
         * The token and its logical position are one indivisible execution input.
         * Implementations must prove that @p logical_state is their current
         * event-published mailbox, wait for its exact producer on the main graph
         * stream, and bind both request-local device pointers to captured replay.
         * The host token is a response/bookkeeping shadow only and must never be
         * uploaded or treated as the execution source of truth.
         *
         * @param token_shadow Host-visible identity of the condition token.
         * @param logical_state Current typed device logical-state publication.
         * @param request_index Request row to consume from the publication.
         * @return true when the main condition graph advances successfully.
         */
        virtual bool advanceMTPMainConditionFromDeviceResidentLogicalState(
            int32_t token_shadow,
            const DeviceResidentLogicalSequenceStateHandle &logical_state,
            int request_index = 0)
        {
            (void)token_shadow;
            (void)logical_state;
            (void)request_index;
            return false;
        }

        /**
         * @brief Advance one GPU main-model condition row from a target sample slot.
         *
         * Implementations must first compose the sampled token and canonical live
         * main-KV position into their resident logical-state mailbox, then consume
         * that typed publication through
         * advanceMTPMainConditionFromDeviceResidentLogicalState(). This contract
         * forbids token-only graph replay and host-derived position scalars.
         *
         * @param token_shadow Host-visible identity of the sampled token.
         * @param target_sample_slot Runner-owned device target-sample slot.
         * @return true when publication and main condition advance both succeed.
         */
        virtual bool advanceMTPMainConditionFromDeviceTargetSample(
            int32_t token_shadow,
            int target_sample_slot)
        {
            (void)token_shadow;
            (void)target_sample_slot;
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
         * forwardGroupedMTPVerifierWithDeviceTokenIds(). The host `token_shadow`
         * still exists for
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
         * @brief Prepare a padded request-batch verifier token matrix on device.
         *
         * Implementations enqueue all first-token and draft-token copies on the
         * verifier graph stream when that stream becomes known, then return a
         * stable runner-owned pointer suitable for forwardBatchWithDeviceTokenIds().
         * The host descriptors are shadows and coordinates only; the verifier
         * embedding graph reads the returned device matrix.
         *
         * @param requests Value-owned row composition descriptors.
         * @param request_count Number of logical rows in the matrix.
         * @param logical_padded_seq_len Largest real row width in this logical
         *        transaction. GPU scalar implementations may bind that logical
         *        prefix into a larger reusable physical capture bucket; the
         *        implementation owns and validates that physical stride.
         * @return Stable device pointer on success, nullptr if the runner cannot
         *         execute the contract exactly.
         */
        virtual const void *prepareMTPVerifierInputTokenBatchOnDevice(
            const DeviceMTPVerifierInputBatchRequest *requests,
            int request_count,
            int logical_padded_seq_len)
        {
            (void)requests;
            (void)request_count;
            (void)logical_padded_seq_len;
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
         * @brief Publish accepted verifier state for a grouped
         *        decode-equivalent outcome.
         *
         * Grouped verifier rows are a hard MTP requirement, not an advertised
         * optional capability.  Implementations must preserve the same
         * semantics as publishAcceptedMTPSpecStateBatch() for the provided step
         * plans while remaining separate from the stronger direct all-position
         * publication policy.
         *
         * @param plans Host-visible publication plans derived from a grouped
         *        decode-equivalent verifier outcome.  The accepted counts and
         *        restore rows must already match the serial replay contract.
         * @param error Optional destination for a human-readable failure reason.
         * @return true if every live-state component was published from the
         *         grouped verifier rows and the runner's host mirrors were left
         *         consistent with the accepted token count.
         */
        virtual bool publishGroupedDecodeEquivalentMTPSpecStateBatch(
            const MTPSpecStepPlanBatch &plans,
            std::string *error = nullptr)
        {
            (void)plans;
            if (error)
                *error =
                    "runner does not support grouped decode-equivalent MTP spec-state publication";
            return false;
        }

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
         * @brief Run a chained MTP sidecar from device-owned token and position state.
         *
         * @param draft_sample_slot Slot in the runner-owned device draft-token
         *        buffer written by sampleStochasticDistributionOnDevice(Draft, ...).
         * @param position_offset Sidecar depth added to the live main KV cache's
         *        canonical device-resident cached-token count.
         * @return true when the graph ran and left logits ready for sampling.
         *
         * The default hard-fails by returning false. GPU implementations must
         * consume both the token slot and live position on device; callers must
         * never substitute a host token upload or host position scalar.
         */
        virtual bool forwardMTPFromDeviceDraftAtLivePositionForDeviceSampling(
            int draft_sample_slot,
            int position_offset)
        {
            (void)draft_sample_slot;
            (void)position_offset;
            return false;
        }

        /**
         * @brief Run the first MTP sidecar from device-owned token and position state.
         *
         * GPU decoding samples the first token into runner-owned device memory.
         * This entry point feeds that token directly to the sidecar embedding
         * and derives its position from the live main KV cache's canonical
         * device count. The default hard-fails; GPU implementations must order
         * both device producers before captured replay.
         */
        virtual bool forwardMTPFromDeviceTargetAtLivePositionForDeviceSampling(
            int target_sample_slot)
        {
            (void)target_sample_slot;
            return false;
        }

        /**
         * @brief Run the first MTP sidecar from a target slot and sample a draft slot.
         *
         * This is the fused greedy companion to
         * forwardMTPFromDeviceTargetAtLivePositionForDeviceSampling(). The
         * first main-model token and the live sidecar position are both
         * device-owned. The MTP draft proposal is sampled into
         * @p draft_sample_slot for the verifier input row. The optional host
         * shadow in @p out_token is for response planning only; the verifier
         * source of truth remains the device draft slot.
         */
        virtual bool forwardMTPFromDeviceTargetAtLivePositionAndSampleGreedyToDeviceDraftSlot(
            int target_sample_slot,
            int draft_sample_slot,
            int32_t *out_token)
        {
            if (!forwardMTPFromDeviceTargetAtLivePositionForDeviceSampling(
                    target_sample_slot))
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
         * @brief Run a resident first sidecar and sample into a device draft slot.
         *
         * This greedy convenience entry point preserves the device mailbox as
         * the sole condition-token and position owner, then samples the sidecar
         * logits into @p draft_sample_slot. A nullable @p out_token is only a
         * response/history shadow; subsequent sidecars and the verifier must
         * consume the device slot.
         */
        virtual bool
        forwardMTPFromDeviceResidentLogicalStateAndSampleGreedyToDeviceDraftSlot(
            const DeviceResidentLogicalSequenceStateHandle &logical_state,
            int request_index,
            int draft_sample_slot,
            int32_t *out_token)
        {
            if (!forwardMTPFromDeviceResidentLogicalStateForDeviceSampling(
                    logical_state,
                    request_index))
            {
                return false;
            }
            return sampleGreedyFromMTPLogitsToDeviceDraftSlot(
                draft_sample_slot,
                out_token);
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
         * consume it without re-uploading the host shadow.  A null
         * @p out_token means the caller has a device-resident verifier/outcome
         * path and is intentionally deferring the D2H token materialization.
         */
        virtual bool forwardMTPAndSampleGreedyToDeviceDraftSlot(
            int32_t draft_condition_token,
            int draft_sample_slot,
            int32_t *out_token)
        {
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
         * @brief Advance one or more resident requests through their condition row.
         *
         * A grouped MTP transaction begins at the same boundary as scalar
         * decode: the main graph first consumes the last token already returned
         * to each request, then samples the transaction's first new target
         * token. GPU implementations must perform the complete transition on
         * device:
         *
         * 1. append one shifted-MTP KV row per request from the current terminal
         *    hidden rows;
         * 2. run one grouped main-model decode from the mailbox condition tokens
         *    and logical positions;
         * 3. sample one target token per row into persistent target slots; and
         * 4. republish the mailbox with positions advanced by one and those
         *    sampled tokens as the next MTP conditions.
         *
         * @param logical_state Live device-owned mailbox before the condition
         *        forward. The handle must cover every request in the batch.
         * @param request_batch Number of active request rows. A value of one is
         *        the canonical SingleDevice transaction; larger values use the
         *        same device-owned contract for continuous request batching.
         * @param params Sampling policy for the newly produced main logits.
         * @param stochastic_position_seeds Optional immutable seed row. It is
         *        required for non-greedy sampling and ignored for greedy
         *        sampling. Mutable positions remain exclusively in
         *        @p logical_state.
         * @return true when the advanced mailbox and target slots are ready.
         *
         * The default hard-fails. There is no scalar, host-token, or stale-logit
         * compatibility path because any such path changes the MTP transaction
         * boundary and breaks decode equivalence.
         */
        virtual bool advanceMTPRequestBatchConditionOnDevice(
            const DeviceResidentLogicalSequenceStateHandle &logical_state,
            int request_batch,
            const SamplingParams &params,
            const uint64_t *stochastic_position_seeds = nullptr)
        {
            (void)logical_state;
            (void)request_batch;
            (void)params;
            (void)stochastic_position_seeds;
            return false;
        }

        /**
         * @brief Run first-depth request-batched MTP from resident logical state.
         *
         * GPU request batching starts from a publication-owned condition-token
         * and position row. Implementations must consume both arrays through
         * @p logical_state, execute one true grouped sidecar, and write proposal
         * `i` to `first_draft_slot + i * slot_stride`. No host token or position
         * shadow is accepted by this contract, so a caller cannot accidentally
         * turn device publication into a D2H/H2D planning loop.
         *
         * @p request_batch must be positive. A value of one is the canonical
         * SingleDevice and scalar LocalTP transaction; larger values exercise
         * the identical kernel and mailbox contract for continuous batching.
         *
         * The default hard-fails. CPU request batching uses the separate
         * forwardMTPBatchAndSampleGreedy() host contract.
         */
        virtual bool forwardMTPBatchFromDeviceResidentLogicalStateAndSampleGreedyToDeviceDraftSlots(
            const DeviceResidentLogicalSequenceStateHandle &logical_state,
            int request_batch,
            int first_draft_slot,
            int slot_stride)
        {
            (void)logical_state;
            (void)request_batch;
            (void)first_draft_slot;
            (void)slot_stride;
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
         * @brief Chain a grouped MTP sidecar directly from device draft slots.
         *
         * Proposal tokens remain in the request-major device matrix written by
         * the previous depth. Implementations gather the strided source column,
         * add @p position_offset to the resident base-position row, execute one
         * grouped sidecar, and publish the next proposal column. Every operation
         * is stream ordered on device; there is deliberately no host-token output.
         *
         * @param logical_state Live resident mailbox that owns base positions.
         * @param request_batch Positive request-row count. One is the canonical
         *        scalar transaction and must not select a separate implementation.
         * @param first_condition_slot Previous depth's first request slot.
         * @param condition_slot_stride Element stride between request inputs.
         * @param position_offset Offset from the mailbox base position for this depth.
         * @param first_draft_slot Destination slot for request zero.
         * @param draft_slot_stride Element stride between request outputs.
         */
        virtual bool forwardMTPBatchFromDeviceDraftSlotsAndSampleGreedyToDeviceDraftSlots(
            const DeviceResidentLogicalSequenceStateHandle &logical_state,
            int request_batch,
            int first_condition_slot,
            int condition_slot_stride,
            int position_offset,
            int first_draft_slot,
            int draft_slot_stride)
        {
            (void)logical_state;
            (void)request_batch;
            (void)first_condition_slot;
            (void)condition_slot_stride;
            (void)position_offset;
            (void)first_draft_slot;
            (void)draft_slot_stride;
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
         * device slot the verifier source of truth.  A null @p out_token is
         * valid only when a later device-resident verifier/outcome reducer will
         * produce the response token list.
         */
        virtual bool forwardMTPFromLastDraftAndSampleGreedyToDeviceDraftSlot(
            int32_t draft_condition_token,
            int position_id,
            int draft_sample_slot,
            int32_t *out_token)
        {
            if (!forwardMTPFromLastDraft(draft_condition_token, position_id))
                return false;
            return sampleGreedyFromMTPLogitsToDeviceDraftSlot(
                draft_sample_slot,
                out_token);
        }

        /**
         * @brief Chained device-slot sidecar plus greedy sample into a device slot.
         *
         * This is the fixed-depth greedy hot-path companion to
         * forwardMTPFromDeviceDraftAtLivePositionForDeviceSampling(). The
         * previous draft token is read from @p draft_condition_sample_slot in
         * runner-owned device memory. The chained sidecar executes at the live
         * device KV count plus @p position_offset, and writes the next proposal
         * to @p draft_sample_slot. The optional host shadow in @p out_token is
         * deliberately nullable; when it is null, callers must consume the
         * compact verifier outcome rather than inspecting `draft_tokens`.
         */
        virtual bool forwardMTPFromDeviceDraftAtLivePositionAndSampleGreedyToDeviceDraftSlot(
            int draft_condition_sample_slot,
            int position_offset,
            int draft_sample_slot,
            int32_t *out_token)
        {
            if (!forwardMTPFromDeviceDraftAtLivePositionForDeviceSampling(
                    draft_condition_sample_slot,
                    position_offset))
            {
                return false;
            }
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
         * @brief Consume a replicated main-logits publication without sampling it.
         *
         * A replicated LocalTP decode graph produces the same full-vocabulary
         * logits row on every participant. Rank-level sampling economically runs
         * the argmax or stochastic sampler only on the primary participant.
         * Every non-primary participant must still close its one-shot producer
         * handoff before the next graph replay; otherwise a later producer could
         * silently replace the stream that owned the previous row.
         *
         * This operation does not synchronize, copy, or launch a replacement
         * sampler. The durable forward-output event remains the ordering owner for
         * later graph/state mutation. Implementations return false unless they
         * explicitly support this replicated-output lifecycle.
         *
         * @return true when no deferred stream exists or the pending replicated
         *         publication was consumed successfully.
         */
        virtual bool consumeUnusedReplicatedMainLogitsPublication()
        {
            return false;
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
         * @brief Append one shifted MTP KV row from a checkpoint's terminal hidden.
         *
         * Grouped verifier publication proves the accepted state after the
         * verifier forward has already produced newer hidden rows.  At that
         * point `commitMTPShiftedRowFromCurrentTerminalHidden()` is too broad:
         * it is allowed to refresh PREFIX_TERMINAL_HIDDEN from the latest
         * verifier tensor, which can publish a shifted row for token[i] using
         * token[i + 1]'s hidden source.  This helper imports only the terminal
         * hidden payload carried by `checkpoint`, leaves KV/GDN/position state
         * untouched, and then appends the shifted sidecar row from that explicit
         * base row.
         *
         * `position_offset_override`, when supplied, must describe the same
         * logical base token count as `checkpoint.cached_tokens`.  A mismatch is
         * a caller bug because the hidden row and shifted-KV append position
         * would refer to different serial-decode boundaries.
         *
         * @param checkpoint Prefix checkpoint captured before speculative MTP
         *        sidecar/verifier work. It must contain a terminal-hidden payload
         *        for the runner or participant that receives it.
         * @param token Token whose depth-0 shifted MTP row should be appended.
         * @param already_appended_tokens Number of shifted rows for this logical
         *        token sequence that already exist before this append.
         * @param allow_speculative_discard Whether extra speculative shifted rows
         *        may be truncated before appending the checkpoint-backed row.
         * @param position_offset_override Optional expected main cached-token
         *        count for the checkpoint boundary.
         */
        virtual bool commitMTPShiftedRowFromCheckpointTerminalHidden(
            const PrefixStateSnapshot &checkpoint,
            int32_t token,
            int already_appended_tokens,
            bool allow_speculative_discard = false,
            int position_offset_override = -1)
        {
            (void)checkpoint;
            (void)token;
            (void)already_appended_tokens;
            (void)allow_speculative_discard;
            (void)position_offset_override;
            return false;
        }

        /**
         * @brief Append the first shifted MTP row from checkpoint hidden and resident outcome token zero.
         *
         * Grouped GPU publication keeps the compact verifier outcome on device
         * until accepted live state is published.  When the sidecar cannot prove
         * that its own row-zero shifted KV append is serial-decode equivalent,
         * the initial shifted row must be rebuilt from the verifier-base
         * terminal hidden checkpoint and the compact outcome's first output
         * token.  Implementations must read that token from
         * @p outcome.output_tokens_device, ordered by @p outcome.response_ready_event;
         * they must not materialize the compact outcome on host just to learn
         * the token or accepted-state count.
         *
         * The method may run the fixed one-row sidecar append even when the
         * compact accepted-state count is zero.  The later device-resident
         * shifted-KV publication derives the serial target length from the same
         * compact metadata and discards the speculative row if it was not
         * accepted.
         *
         * @param checkpoint Verifier-base checkpoint containing terminal hidden
         *        for this runner or participant.
         * @param outcome Device-resident compact verifier outcome.
         * @param request_index Logical request row inside @p outcome.
         * @param main_forward_token_count Verifier hidden-row count to restore
         *        after the sidecar append; pass the grouped verifier row count.
         * @param allow_speculative_discard Whether stale speculative shifted rows
         *        may be truncated before appending row zero.
         * The verifier-base position must come from the same resident metadata
         * workspace captured before verifier replay. A host scalar position is
         * not part of this API.
         */
        virtual bool commitMTPInitialShiftedRowFromDeviceOutcome(
            const PrefixStateSnapshot &checkpoint,
            const DeviceSpeculativeOutcomeHandle &outcome,
            int request_index,
            int main_forward_token_count,
            bool allow_speculative_discard = false)
        {
            (void)checkpoint;
            (void)outcome;
            (void)request_index;
            (void)main_forward_token_count;
            (void)allow_speculative_discard;
            return false;
        }

        /**
         * @brief Append one shifted MTP KV row from a device-resident target token.
         *
         * Penalty-free stochastic GPU decode can defer the first main-token host
         * read.  The initial shifted-cache repair after verifier-base restore
         * must still append that token's row, so supporting runners read the
         * token from the same target sample slot used by
         * forwardMTPFromDeviceTargetAtLivePositionForDeviceSampling().
         *
         * The append position is derived from the request transaction's canonical
         * device-resident shifted-cache count.  A host position argument is
         * intentionally absent: accepting one would let token and position
         * ownership split across independently advancing GPU and CPU timelines.
         */
        virtual bool commitMTPShiftedRowFromDeviceTargetSample(
            int target_sample_slot,
            int already_appended_tokens,
            bool allow_speculative_discard = false)
        {
            (void)target_sample_slot;
            (void)already_appended_tokens;
            (void)allow_speculative_discard;
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
         * The target position is part of @p logical_state and must be consumed
         * directly on the device. Implementations must never request a scalar
         * host position override or derive one from get_position() /
         * sequence_lengths(); doing so would split one resident transaction
         * across two independently advancing owners.
         *
         * This is a consuming operation. A successful shifted-KV mutation
         * advances the live-state epoch and must retire the supplied logical
         * mailbox rather than retargeting its old values into the new epoch.
         * Callers that need the condition token for the subsequent main graph
         * must first publish it D2D into a persistent target slot.
         */
        virtual bool commitMTPShiftedRowFromDeviceResidentLogicalState(
            const DeviceResidentLogicalSequenceStateHandle &logical_state,
            int request_index,
            int already_appended_tokens,
            bool allow_speculative_discard = false)
        {
            (void)logical_state;
            (void)request_index;
            (void)already_appended_tokens;
            (void)allow_speculative_discard;
            return false;
        }

        /**
         * @brief Append shifted MTP KV suffix rows from a resident verifier outcome.
         *
         * Grouped LocalTP MTP publication keeps compact verifier metadata and
         * output tokens on device until after live state is published.  When the
         * shifted sidecar cache already owns the first accepted row, this method
         * appends the remaining accepted prefix rows by reading
         * @p outcome.output_tokens_device and @p outcome.meta_device on an
         * explicit GPU stream.  Implementations must not materialize the compact
         * outcome on host to learn the accepted count.  They may run a fixed
         * bounded suffix shape and rely on device-resident shifted-KV publication
         * to discard rows beyond the compact accepted-state count.
         *
         * @param outcome Device-resident compact verifier output handle.
         * @param request_index Logical request row inside @p outcome.
         * @param already_appended_tokens Number of accepted shifted rows already
         *        resident, usually one sidecar-owned row.
         * @param max_state_commit_rows Maximum verifier state rows represented by
         *        the outcome's graph shape.
         * @param main_forward_token_count Verifier hidden-row count available for
         *        suffix row selection.
         * @param allow_speculative_discard Whether stale speculative shifted rows
         *        may be truncated before appending the bounded suffix.
         * The shifted cache's resident row count is read from its canonical
         * device metadata; callers do not supply a host count shadow.
         */
        virtual bool commitMTPShiftedRowsFromDeviceOutcome(
            const DeviceSpeculativeOutcomeHandle &outcome,
            int request_index,
            int already_appended_tokens,
            int max_state_commit_rows,
            int main_forward_token_count,
            bool allow_speculative_discard = false)
        {
            (void)outcome;
            (void)request_index;
            (void)already_appended_tokens;
            (void)max_state_commit_rows;
            (void)main_forward_token_count;
            (void)allow_speculative_discard;
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

        /**
         * @brief Consume local verifier logits for an explicit host gather.
         *
         * GPU implementations must join the grouped verifier publication onto
         * an explicit host-bridge stream before returning. This contract keeps
         * diagnostics and parity snapshots ordered without synchronizing the
         * producer stream or retaining graph-cache stream pointers.
         */
        virtual LogitsLocalInfo consumeAllPositionLogitsLocalInfoForHostGather()
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
         * @brief True when MTP verifier graphs use a mirrored terminal head.
         *
         * This is a declarative ownership fact, not a fallback capability. In
         * every TP scope, each participant owns full-vocabulary verifier logits
         * and may reduce its compact outcome locally before publishing from the
         * same device-resident handle. Returning false means the explicitly
         * vocabulary-sharded policy is active and rank-scope candidate
         * coordination is required.
         */
        virtual bool usesMirroredMTPHeadForVerifier() const { return false; }

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
         * When @p out_token is non-null, implementations also return a host
         * shadow for legacy response planning.  When @p out_token is null, the
         * sample is fully deferred and must not perform a D2H copy; the compact
         * verifier outcome is responsible for later host-visible response
         * tokens.  In both cases the runner must write @p draft_sample_slot in
         * the same
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
         * response bridge. Focused parity diagnostics may explicitly copy the
         * resulting compact row, but production generation must keep it resident.
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
         * @brief Configure immutable stop-token controls for the next request.
         *
         * GPU runners must stage these host values as request policy and publish
         * them to a persistent device buffer at request admission or prefix
         * restore. The verifier hot path may validate this policy, but it must
         * never upload the control row per speculative transaction.
         *
         * The default accepts CPU runners, whose sampling policy remains
         * host-owned. A GPU runner that does not implement explicit request
         * control publication fails immediately instead of silently reverting
         * to per-step host involvement.
         *
         * @param stop_tokens Request-constant token IDs that terminate serving.
         * @return True when the runner owns the complete request policy.
         */
        virtual bool configureMTPRequestStopTokens(
            const std::vector<int32_t> &stop_tokens)
        {
            (void)stop_tokens;
            if (primaryDeviceId().is_gpu())
            {
                throw std::logic_error(
                    "GPU inference runner does not implement device-owned MTP "
                    "request stop-token publication");
            }
            return true;
        }

        /**
         * @brief Admit immutable MTP penalty magnitudes for the next request.
         *
         * GPU implementations publish this policy exactly once after the
         * request-state reset (or prefix-restore reset) and before any sampling
         * graph consumes it.  The evolving "first condition is already in
         * history" predicate is deliberately not an argument: accepted-state
         * publication owns that device-resident transition.
         *
         * Reconfiguring the policy after a verifier transaction has started is a
         * lifecycle violation.  Implementations must fail rather than patching a
         * captured graph or silently retaining the previous request's policy.
         */
        virtual bool configureMTPRequestPenaltyPolicy(
            const MTPRequestPenaltyPolicy &policy)
        {
            (void)policy;
            if (primaryDeviceId().is_gpu())
            {
                throw std::logic_error(
                    "GPU inference runner does not implement device-owned MTP "
                    "request penalty-policy publication");
            }
            return true;
        }

        /**
         * @brief Arm the terminal greedy outcome stage before verifier replay.
         *
         * Stop-token controls must already be device-resident from explicit
         * request admission. Draft tokens already reside in the verifier input
         * arena row. Once armed, failure to execute or consume the matching
         * graph transaction is fatal; implementations must not upload controls
         * or enqueue a post-graph reducer as a substitute.
         */
        virtual bool prepareGreedyAllPositionBatchOutcomeGraph(
            int verifier_token_count,
            const int32_t *stop_tokens,
            int stop_token_count,
            const MTPRequestPenaltyPolicy &penalty_policy =
                MTPRequestPenaltyPolicy{})
        {
            (void)verifier_token_count;
            (void)stop_tokens;
            (void)stop_token_count;
            (void)penalty_policy;
            return false;
        }

        /**
         * @brief Summarize a request-batched greedy verifier into resident rows.
         *
         * This is the request-batch version of
         * verifyGreedyAllPositionBatchOutcomeOnDeviceResident().  Callers must
         * have already run one grouped verifier forward using the device matrix
         * prepared by prepareMTPVerifierInputTokenBatchOnDevice().  Implementations
         * enqueue all argmax and greedy summary kernels on the pending verifier
         * stream and return compact outcome rows that
         * publishAcceptedMTPSpecStateBatchFromDeviceOutcome() can consume before
         * any host response bridge runs.
         */
        virtual bool verifyGreedyAllPositionRequestBatchOutcomesOnDeviceResident(
            const DeviceGreedyBatchOutcomeRequest *requests,
            int request_count,
            DeviceSpeculativeOutcomeHandle *out_handle)
        {
            (void)requests;
            (void)request_count;
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
         * @brief Publish resident next-condition tokens to a host result buffer.
         *
         * Device-owned MTP keeps sampled condition tokens in the logical-state
         * mailbox. Tests and server result surfaces occasionally need those
         * compact values on the host, but must not rediscover them by sampling
         * an ambiguous logits tensor. Implementations wait on the mailbox's
         * readiness event and copy only the requested INT32 result rows.
         *
         * This is an explicit result boundary, not an execution input path.
         * GPU implementations must never adopt the copied values back into
         * live state or use a device-wide synchronization.
         *
         * @param logical_state Current typed mailbox handle.
         * @param request_count Number of leading request rows to observe.
         * @param out_tokens Host destination with @p request_count entries.
         * @return true when the compact result was published successfully.
         */
        virtual bool observeDeviceResidentNextConditionTokens(
            const DeviceResidentLogicalSequenceStateHandle &logical_state,
            int request_count,
            int32_t *out_tokens)
        {
            (void)logical_state;
            (void)request_count;
            (void)out_tokens;
            return false;
        }

        /**
         * @brief Get vocabulary size
         */
        virtual int vocab_size() const = 0;

        /**
         * @brief Reset request-scoped inference state before a new prompt/session.
         *
         * This is the named state-lifetime boundary. Callers use it at request
         * boundaries, benchmark iteration boundaries, and after hard
         * replay/restore failures when the next token stream must start from an
         * empty live sequence. The request object names which first-class state
         * owners are crossing the boundary: KV/GDN, MTP, and logical sequence
         * metadata.
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
         * explicitly named API.
         */
        virtual void resetInferenceState(const InferenceStateResetRequest &request)
        {
            if (request.boundary != InferenceStateResetRequest::Boundary::Request ||
                !request.resetsAllLiveRequestOwners() ||
                !request.reset_model_runtime ||
                !request.preserve_replay_safe_graphs)
            {
                throw std::invalid_argument(
                    "IInferenceRunner::resetInferenceState fallback only supports full "
                    "request-boundary resets that preserve replay-safe graph captures");
            }
            clear_cache();
        }

        /**
         * @brief Compatibility wrapper for the historical request-boundary reset.
         *
         * New code should call resetInferenceState() with an explicit
         * InferenceStateResetRequest. This name remains for older tests and
         * adapter interfaces while the API migration proceeds.
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
         *         -1 if not supported or failed. Callers may use host logits
         *         only for CPU-only execution; GPU decode treats this as a
         *         hard failure so it does not silently copy logits to host.
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
         * forwardMTPFromDeviceTargetAtLivePositionForDeviceSampling(). Passing nullptr for
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
         * @brief GPU-side sampling keyed by the logical output position.
         *
         * Seeded stochastic decode must use the same draw for a token whether it
         * is sampled by ordinary serial decode or by an MTP verifier bonus row.
         * Implementations should use @p logical_position as the draw key for the
         * SamplingMath MTP "Sample" purpose. Returning -1 means the runner cannot
         * provide that production sampler path.
         */
        virtual int sampleOnDeviceAtLogicalPosition(
            const SamplingParams &params,
            int logical_position)
        {
            (void)params;
            (void)logical_position;
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
         * @brief Publish request-batched prefill samples into resident state.
         *
         * Request-batched prefill writes one terminal logits row per logical
         * request. A GPU implementation samples every row, stores each token in
         * its persistent target slot, and publishes the corresponding logical
         * position, sequence length, and next-condition rows through one exact
         * producer event. No sampled token is returned to the host: the first
         * grouped verifier transaction consumes these rows directly and the
         * complete generation parent is the sole owner of response materialization.
         *
         * @param request_count Number of active request rows to publish.
         * @param params Sampling parameters shared by the request batch.
         * @param stochastic_position_seeds Optional immutable request seeds
         *        [request_count] for non-greedy sampling. The runner combines
         *        each seed with its device-resident logical position; callers
         *        must never compute or pass host threshold values. A non-greedy
         *        request requires one non-zero seed per row.
         * @return true only after every row and its publication event are valid.
         */
        virtual bool publishMainLogitsBatchSamplesToDeviceResidentState(
            int request_count,
            const SamplingParams &params,
            const uint64_t *stochastic_position_seeds = nullptr)
        {
            (void)request_count;
            (void)params;
            (void)stochastic_position_seeds;
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
         * @brief Apply decode-boundary maintenance after a successful committed step.
         *
         * Implementations that own graph-captured MoE maintenance launch it
         * here, after MTP verifier publication or rollback has closed.  Raw
         * forward execution is deliberately not a substitute for this hook.
         */
        virtual bool maybeApplyDecodeBoundaryMaintenance() { return true; }

        /**
         * @brief Select the exact maintenance scheduler before a boundary.
         *
         * Rank schedulers require every collective participant to report the
         * same non-inactive policy. A homogeneous CUDA domain must report the
         * native conditional graph; a homogeneous ROCm domain must report the
         * authenticated hosted-ticket policy until HIP gains equivalent graph
         * nodes. The default is inactive for CPU and non-MoE runners.
         */
        virtual DeviceMoERebalanceMaintenanceExecutionPolicy
        deviceMoERebalanceMaintenanceExecutionPolicy() const noexcept
        {
            return DeviceMoERebalanceMaintenanceExecutionPolicy::Inactive;
        }

        /**
         * @brief Return immutable bounds for economical hosted observation.
         *
         * Hosted ROCm implementations must return one valid schedule shared by
         * every collective participant. Other policies return an invalid empty
         * schedule because they never use a host ticket cadence.
         */
        virtual DeviceMoERebalanceHostedObservationSchedule
        deviceMoERebalanceHostedObservationSchedule() const noexcept
        {
            return {};
        }

        /**
         * @brief Validate one provably non-due HIP boundary transaction.
         *
         * Ordinary serial decode owns publish/ack inside its complete captured
         * graph, while grouped MTP owns publication in accepted-state commit
         * and acknowledgement at its next admission. This hook verifies that
         * graph topology was installed and records scheduler evidence; it must
         * launch no graph, record no event, release no ExpertOverlay reader,
         * and perform no D2H. Callers may use it only while the immutable
         * hosted schedule proves the current boundary cannot be due.
         */
        virtual bool
        submitHostScheduledDeviceMoERebalanceKnownNonDueBoundary()
        {
            return false;
        }

        /**
         * @brief Publish and observe one fresh HIP maintenance ticket.
         *
         * Implementations join the committed inference timeline, replay the
         * captured ticket publisher over the already-published device edge,
         * and copy only
         * @ref DeviceMoERebalanceDispatchTicket to host. They retain the current
         * ExpertOverlay reader until an authenticated due decision establishes
         * that a placement writer will run. They do not submit maintenance or
         * acknowledgement until the rank has compared every participant's
         * decision.
         */
        virtual bool observeDeviceMoERebalanceDispatchTicket(
            DeviceMoERebalanceDispatchTicket *out_ticket)
        {
            (void)out_ticket;
            return false;
        }

        /**
         * @brief Submit the rank-authenticated HIP maintenance decision.
         *
         * A due ticket submits the retained collective maintenance graph. A
         * pending non-due MTP ticket submits the retained acknowledgement
         * graph, while an already-acknowledged serial-decode ticket performs no
         * graph work. The call must byte-match this runner's last observation
         * and performs no host state upload or additional ticket read.
         */
        virtual bool submitHostScheduledDeviceMoERebalanceMaintenance(
            const DeviceMoERebalanceDispatchTicket &ticket)
        {
            (void)ticket;
            return false;
        }

        /**
         * @brief Drain completed decode-boundary maintenance diagnostics.
         *
         * Benchmark and serving request epilogues use this to retire
         * asynchronous maintenance events after measured decode has finished,
         * without forcing a full request reset or pushing diagnostics into the
         * next request.
         */
        virtual void drainCompletedDecodeBoundaryMaintenanceDiagnostics() {}

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
         * @return true if applied on device, false if unsupported or failed.
         *         GPU decode callers must treat false as a hard failure rather
         *         than silently falling back to host logits.
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
         * @brief Apply device-owned generated-history penalties to GPU logits.
         *
         * `Main` applies only the durable generated-token histogram to the next
         * target row. `AllPosition` also folds the device-resident verifier
         * input prefix into each row, giving row `r` exactly the history seen by
         * serial decode. The operation is stream ordered and performs no host
         * transfer, allocation, atomics, or synchronization.
         *
         * DRY is intentionally outside this count-histogram policy. Callers
         * must reject DRY before entering this method until a device-owned
         * sequence-history implementation is selected.
         */
        virtual bool applyDeviceOwnedMTPPenaltiesToLogitRows(
            DeviceLogitsSource source,
            int row_count,
            const MTPRequestPenaltyPolicy &penalty_policy)
        {
            (void)source;
            (void)row_count;
            (void)penalty_policy;
            return false;
        }

        /**
         * @brief Apply the current device-resident MTP proposal branch history.
         *
         * The sidecar's first condition token and slots `[0,
         * prior_draft_count)` are the only speculative tokens visible while
         * scoring the next draft proposal.  Implementations must consume those
         * persistent device owners on the sidecar producer stream and must not
         * materialize a host token shadow or upload a sparse penalty map.
         */
        virtual bool applyDeviceOwnedMTPBranchPenaltiesToLogits(
            int prior_draft_count,
            const MTPRequestPenaltyPolicy &penalty_policy)
        {
            (void)prior_draft_count;
            (void)penalty_policy;
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
         * @brief Build every stochastic verifier target row as one captured transaction.
         *
         * The operation consumes the current all-position verifier logits and
         * produces compact target distributions in slots `[0, row_count)`.
         * Device-owned history penalties, when enabled by @p penalty_policy,
         * are part of the same monolithic graph as top-k/top-p construction.
         * Implementations must reject missing graph capture, segmented replay,
         * null producer streams, stale arena bindings, and non-GPU execution;
         * eager execution is not a permitted substitute.
         *
         * @param row_count Number of contiguous verifier rows, including bonus.
         * @param params Immutable stochastic sampling policy captured by graph.
         * @param penalty_policy Device-history penalty policy for these rows.
         * @param vocab_size Expected full vocabulary width of each logits row.
         * @return true only after strict graph replay has been queued and its
         *         completion handed back to the verifier producer stream.
         */
        virtual bool buildCapturedStochasticVerifierTargetDistributions(
            int row_count,
            const SamplingParams &params,
            const MTPRequestPenaltyPolicy &penalty_policy,
            int vocab_size)
        {
            (void)row_count;
            (void)params;
            (void)penalty_policy;
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
         * @brief Publish one MTP proposal through a mandatory captured graph.
         *
         * The method consumes the exact sidecar-logit producer, applies the
         * supplied serial branch-penalty policy when enabled, and records a
         * device-resident draft-slot readiness edge.  GPU implementations may
         * not substitute eager execution or return a host token.
         */
        virtual bool publishCapturedMTPDraftToken(
            int row,
            int slot,
            const MTPRequestPenaltyPolicy &penalty_policy)
        {
            (void)row;
            (void)slot;
            (void)penalty_policy;
            return false;
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
         * @brief Acquire a ready target-sample mailbox as a collective source.
         *
         * The source stream is the exact target sampler stream and the slot must
         * already own a sample-ready publication.
         */
        virtual DeviceStochasticTargetSampleSlotHandle
        deviceStochasticTargetSampleProducerSlot(int slot)
        {
            (void)slot;
            return {};
        }

        /**
         * @brief Acquire a peer target mailbox after its main-forward producer.
         *
         * Implementations consume the peer's replicated main-logits publication
         * and join its durable main-forward event before returning the collective
         * destination stream.  This keeps terminal hidden, live model state, and
         * the broadcast token on one explicit device timeline.
         */
        virtual DeviceStochasticTargetSampleSlotHandle
        deviceStochasticTargetSampleBroadcastDestinationSlot(int slot)
        {
            (void)slot;
            return {};
        }

        /**
         * @brief Publish a device collective as one target slot's producer.
         *
         * Implementations record a fresh readiness event after the collective
         * write so verifier token materialization waits on the common rank-owned
         * target rather than a pre-broadcast child-local sampler event.
         */
        virtual bool recordStochasticTargetSampleSlotReadyFromDevice(
            int slot,
            void *producer_stream,
            bool verifier_consumer_pending = true)
        {
            (void)slot;
            (void)producer_stream;
            (void)verifier_consumer_pending;
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

        /**
         * @brief Publish one host-resolved control token into a device target slot.
         *
         * LocalTP rank-level stochastic sampling reduces per-shard compact
         * candidates into one full-vocab token.  Once that token is known, each
         * child runner still needs it in the same runner-owned device slot used
         * by native device samplers, so later sidecar and verifier graph inputs
         * can consume a stable device pointer instead of a host token row.
         * Request-policy tokens, such as bounded-thinking stop-sequence tokens,
         * use the same ownership boundary before entering MTP state.
         *
         * Implementations must enqueue a scalar publication kernel on an explicit
         * backend stream and record the usual target-sample readiness event.
         * They must not issue an asynchronous H2D copy from the caller's stack
         * scalar. Returning false is a hard capability failure for callers that
         * selected the device-token path.
         */
        virtual bool stageStochasticTargetTokenForDeviceSampling(
            int32_t target_token,
            int target_sample_slot = 0)
        {
            (void)target_token;
            (void)target_sample_slot;
            return false;
        }

        /**
         * @brief Publish a resident condition token into a device target slot.
         *
         * A verifier publication mailbox can outlive the transaction that
         * produced it long enough to seed the next decode step. Some consumers,
         * notably budget-limited direct emit, must both mutate the mailbox-owned
         * shifted-MTP transaction and retain the same token for a later main
         * graph replay. Implementations copy the request-local
         * `next_condition_tokens_device` entry into the persistent target-sample
         * arena before that mutation, using an explicit stream ordered after the
         * mailbox readiness event, and publish the normal target-slot readiness
         * event after the copy.
         *
         * This operation is strictly device-to-device. It must never read the
         * host token shadow or call a host-to-device transfer API. A stale,
         * foreign, out-of-range, or streamless mailbox is a hard contract
         * failure for GPU callers.
         *
         * @param logical_state Runner-owned resident publication mailbox.
         * @param request_index Request row containing the condition token.
         * @param target_sample_slot Persistent target slot receiving the token.
         * @return true only when the D2D copy and target readiness publication
         *         were enqueued successfully.
         */
        virtual bool publishDeviceResidentConditionTokenToTargetSampleSlot(
            const DeviceResidentLogicalSequenceStateHandle &logical_state,
            int request_index,
            int target_sample_slot = 0)
        {
            (void)logical_state;
            (void)request_index;
            (void)target_sample_slot;
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
         * scheduler-oriented path. Production publication consumes the returned
         * handle directly; only focused parity diagnostics may materialize it.
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
            request.draw_position_source =
                derive_thresholds_from_seed
                    ? DeviceStochasticDrawPositionSource::HostLogicalPosition
                    : DeviceStochasticDrawPositionSource::ExplicitThresholds;
            request.use_device_draft_tokens = draft_tokens == nullptr;
            if (!request.ensureHostRowCapacity(row_count))
                return false;

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
            request.draw_position_source =
                derive_thresholds_from_seed
                    ? DeviceStochasticDrawPositionSource::HostLogicalPosition
                    : DeviceStochasticDrawPositionSource::ExplicitThresholds;
            request.use_device_draft_tokens = draft_tokens == nullptr;
            if (!request.ensureHostRowCapacity(row_count))
                return false;

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
                const size_t required_rows =
                    static_cast<size_t>(request.row_count);
                if ((!request.use_device_draft_tokens &&
                     request.draft_tokens.size() < required_rows) ||
                    (!request.derive_thresholds_from_seed &&
                     (request.accept_thresholds.size() < required_rows ||
                      request.residual_thresholds.size() < required_rows)))
                {
                    return false;
                }
                if (request.serial_sample_equivalent)
                {
                    /*
                     * The serial-sample-equivalent contract is device-resident:
                     * implementations must compare sampled target tokens against
                     * the same verifier-token row consumed by the grouped graph.
                     * The conservative default cannot synthesize that row without
                     * falling back to host replay, so fail loudly.
                     */
                    return false;
                }
                if (request.derive_thresholds_from_seed &&
                    request.draw_position_source !=
                        DeviceStochasticDrawPositionSource::HostLogicalPosition)
                {
                    /*
                     * A generic runner has no device-position authority.  It
                     * must never turn a resident-position descriptor back into
                     * host scalar arrays; concrete GPU runners consume those
                     * descriptors directly on their explicit streams.
                     */
                    return false;
                }
                std::vector<float> derived_accept_thresholds(
                    static_cast<size_t>(request.row_count),
                    0.0f);
                std::vector<float> derived_residual_thresholds(
                    static_cast<size_t>(request.row_count),
                    0.0f);
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
         * output rows. This is the GPU-resident production contract. Intermediate
         * response metadata must remain resident until the terminal ledger is
         * surfaced; only focused diagnostics may inspect an individual outcome.
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
         * @brief Admit one grouped generation response ledger on the device.
         *
         * GPU implementations initialize persistent response-token and control
         * rows exactly once after prefill.  Every later verifier transaction
         * consumes and republishes that controller through explicit event
         * edges; no host counter becomes an alternate commit-boundary owner.
         * CPU and runners without a resident stochastic path may retain the
         * default no-op implementation.
         *
         * @param request Typed immutable geometry, response capacity, and
         *        leading-row response ownership for this admission.
         * @return true when generation may begin.
         */
        virtual bool beginDeviceResidentGeneration(
            const DeviceGenerationAdmissionRequest &request)
        {
            return request.valid();
        }

        /**
         * @brief Select the complete-loop execution policy before admission.
         *
         * Production device and rank orchestrators override this method using
         * their live graph-capture capabilities and participant topology. The
         * interface default is deliberately unsupported: a device identifier
         * alone cannot prove that a complete graph family was materialized.
         *
         * @param topology Fixed or device-selected draft-depth topology.
         * @return One explicit policy. `Unsupported` must fail before request
         *         admission rather than selecting another path after failure.
         */
        virtual DeviceGenerationExecutionPolicy
        deviceGenerationExecutionPolicy(
            DeviceGenerationLoopTopology topology) const noexcept
        {
            (void)topology;
            return DeviceGenerationExecutionPolicy::Unsupported;
        }

        /**
         * @brief Compose the exact policy-complete device generation executable.
         *
         * The first externally orchestrated transaction must already have
         * committed its resident response/state rows, and every child graph in
         * the family must already own a strict monolithic executable. In MoE
         * domains this method also owns the first completed maintenance
         * boundary: every rank participant launches that boundary on its
         * persistent worker before cloning the now-materialized maintenance
         * child into the parent. Keeping the bootstrap inside the runner makes
         * it impossible for host orchestration to become a second placement
         * scheduler.
         *
         * Rank implementations must complete this preparation for every local
         * participant before any participant launches. The method performs the
         * first asynchronous maintenance publication and graph composition; it
         * must not launch generation, synchronize a stream/device, materialize
         * live state on the host, or recover through segmented/eager execution.
         *
         * @param request_count Number of admitted resident controller rows.
         * @param draft_depth Fixed depth, or maximum capture depth for a dynamic
         *        child family. The verifier child owns `draft_depth + 1` rows.
         * @param topology Typed fixed/dynamic depth topology already selected
         *        before admission. CUDA embeds it in a conditional parent;
         *        HIP retains every legal branch and captures only its immutable
         *        dispatch-ticket publisher.
         * @param sampling_mode Exact compact-outcome topology embedded in every
         *        transaction body. It is part of graph-cache identity.
         * @return true when the complete policy executable is ready to launch.
         */
        virtual bool materializeDeviceResidentGeneration(
            int request_count,
            int draft_depth,
            DeviceGenerationLoopTopology topology,
            DeviceGenerationSamplingMode sampling_mode)
        {
            (void)request_count;
            (void)draft_depth;
            (void)topology;
            (void)sampling_mode;
            return false;
        }

        /**
         * @brief Execute the complete device-owned generation policy.
         *
         * The first transaction has already committed its compact outcome and
         * materialized the exact child graph family. CUDA enqueues one native
         * conditional parent asynchronously. HIP advances retained captured
         * transactions from authenticated device-published tickets; only the
         * outer launch decision is host-visible, while mutable state and the
         * dynamic depth controller remain device-owned. This method may be
         * called exactly once for an admitted request.
         */
        virtual bool launchDeviceResidentGeneration()
        {
            return false;
        }

        /**
         * @brief Observe one authenticated device-owned graph-dispatch decision.
         *
         * This operation exists only for the explicit hosted-transaction
         * policy used by HIP conditional-graph emulation and heterogeneous
         * sparse-collective boundaries. Implementations publish a fixed ticket on device, enqueue one
         * ticket-only D2H copy, and wait on that copy's exact event. They must
         * never materialize a compact verifier outcome or any mutable inference
         * state. Rank schedulers call this method for every participant and
         * compare decisions before submitting another collective-bearing branch.
         *
         * @param out_ticket Destination for the immutable scheduling snapshot.
         * @return true when one fresh, lifecycle-authenticated ticket was read.
         */
        virtual bool observeDeviceGenerationDispatchTicket(
            sampling_math::DeviceGenerationDispatchTicket *out_ticket)
        {
            (void)out_ticket;
            return false;
        }

        /**
         * @brief Validate one ticket and expose its ordered hosted fragments.
         *
         * A rank scheduler calls this for every participant before submitting
         * any fragment. Implementations must authenticate the last observed
         * ticket, retire the prior sparse graph sequence at its exact ticket
         * fence, and open the selected next sequence without launching work.
         * The returned count includes due maintenance and is zero only for a
         * terminal ticket with no maintenance.
         *
         * @param ticket Last authenticated device dispatch decision.
         * @param out_fragment_count Number of fragments selected by the ticket.
         * @return True when a non-overlapping advance was opened.
         */
        virtual bool beginHostScheduledDeviceGenerationAdvance(
            const sampling_math::DeviceGenerationDispatchTicket &ticket,
            size_t *out_fragment_count)
        {
            (void)ticket;
            if (out_fragment_count)
                *out_fragment_count = 0;
            return false;
        }

        /**
         * @brief Enqueue one ordered fragment of an opened hosted advance.
         *
         * Rank orchestration submits the same fragment ordinal concurrently on
         * every LocalTP participant, then waits only for those host submissions
         * to return before moving to the next ordinal. This preserves symmetric
         * sparse graph-group entry without synchronizing device execution.
         */
        virtual bool submitHostScheduledDeviceGenerationFragment(
            const sampling_math::DeviceGenerationDispatchTicket &ticket,
            size_t fragment_index)
        {
            (void)ticket;
            (void)fragment_index;
            return false;
        }

        /**
         * @brief Seal one fully submitted hosted advance.
         *
         * Live tickets enqueue the next immutable ticket observation. Terminal
         * tickets publish terminal device-state readiness. The method performs
         * no device synchronization and rejects missing or duplicate fragments.
         */
        virtual bool finishHostScheduledDeviceGenerationAdvance(
            const sampling_math::DeviceGenerationDispatchTicket &ticket)
        {
            (void)ticket;
            return false;
        }

        /**
         * @brief Submit the branch selected by the last observed hosted ticket.
         *
         * The default remains unsupported. Production device/rank runners use
         * the begin/fragment/finish protocol above so a heterogeneous retained
         * branch can interleave symmetric participant submissions at each
         * sparse-collective boundary. The call is asynchronous with respect to
         * device execution and never uploads mutable inference state.
         *
         * @param ticket Last ticket returned by this runner.
         * @return true when the next device work was submitted successfully.
         */
        virtual bool submitHostScheduledDeviceGenerationAdvance(
            const sampling_math::DeviceGenerationDispatchTicket &ticket)
        {
            (void)ticket;
            return false;
        }

        /**
         * @brief Surface and close one completed device-owned generation.
         *
         * GPU implementations consume the final controller-ready event on a
         * dedicated result stream, enqueue the response and controller copies,
         * record one terminal event, wait for that exact event, and validate the
         * complete controller ABI before releasing the request lifecycle. Calling this
         * method before every request is terminal is an error, not a polling API.
         */
        virtual bool finishDeviceResidentGeneration(
            DeviceGenerationTerminalResult *out_result)
        {
            (void)out_result;
            return false;
        }

        /**
         * @brief Copy one compact outcome to host for a focused diagnostic.
         *
         * This operation is an oracle/probe boundary for parity and kernel tests.
         * It may synchronize a dedicated diagnostic stream and is therefore
         * forbidden in production generation. Production state publication and
         * response construction consume DeviceSpeculativeOutcomeHandle directly
         * and surface only the final terminal ledger.
         */
        virtual bool copyDeviceSpeculativeOutcomesToHostForDiagnostics(
            const DeviceSpeculativeOutcomeHandle &handle,
            DeviceSpeculativeVerifyBatchOutcome *outcomes)
        {
            (void)handle;
            (void)outcomes;
            return false;
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
         * @brief Restrict snapshot capture to a set of published snapshot keys.
         *
         * Empty means capture every instrumented stage, preserving the legacy
         * snapshot-infrastructure behavior. Graph-captured parity tests use this
         * to avoid allocating point-in-time device copies for irrelevant stages.
         */
        virtual void setSnapshotCaptureFilter(const std::vector<std::string> &keys)
        {
            (void)keys;
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
         * @brief Consume local main logits for an explicit host gather.
         *
         * Metadata returned by getLogitsLocalInfo() does not establish an
         * ordering edge from a GPU graph producer to a host transfer. GPU
         * runners override this method to consume that producer publication
         * onto a dedicated, non-null host-bridge stream. LogitsGatherer rejects
         * GPU metadata without such a stream, making unordered D2H impossible.
         *
         * CPU runners may use the metadata-only implementation because their
         * tensor storage is already host resident.
         *
         * @return Ordered local-logits information, or empty when unavailable.
         */
        virtual LogitsLocalInfo consumeLogitsLocalInfoForHostGather()
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

        /**
         * @brief Consume local MTP logits for an explicit host gather.
         *
         * GPU implementations must return the exact host-bridge stream ordered
         * after the sidecar graph. Returning a metadata-only GPU view is not a
         * valid implementation and is rejected by LogitsGatherer.
         */
        virtual LogitsLocalInfo consumeMTPLogitsLocalInfoForHostGather()
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
         * @brief Domain-local MoE runtime movement epoch.
         *
         * This is the observable "expert placement data changed" epoch. For
         * CPU/host-applied rebalancing it normally matches moePlacementEpoch().
         * Graph-stable GPU rebalancing keeps moePlacementEpoch() out of graph
         * cache keys and increments this value when device-side runtime tables
         * or transfer-slot backed residency state are updated.
         */
        virtual uint64_t moeRuntimeMovementEpoch() const { return moePlacementEpoch(); }

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
         * @brief Lookup a MoE rebalance controller by routed-expert domain id.
         */
        virtual MoERebalanceController *moeRebalanceControllerForDomain(
            const std::string &domain_id) const
        {
            (void)domain_id;
            return nullptr;
        }

        /**
         * @brief Return where the sole ExpertOverlay authority executes.
         *
         * This is topology, not an optional capability bit. A runner that
         * participates in an ExpertOverlay graph returns the frozen selection
         * carried by its graph configuration. Composite runners must require
         * every participant to report the same selection; disagreement is an
         * invalid graph family rather than permission to choose a fallback.
         */
        virtual MoEOverlayAuthorityExecutionKind
        moeOverlayAuthorityExecution() const
        {
            return MoEOverlayAuthorityExecutionKind::Unresolved;
        }

        /**
         * @brief Confirm that the selected device authority owns executable topology.
         *
         * This setup-time query is true only for an active dynamic authority
         * whose required captured maintenance graph, persistent workspace
         * generation, collective participant, and epoch arena were
         * materialized. It is not a runtime fallback or a feature probe: a
         * false result after dynamic device-resident selection is a fatal
         * model-construction error. Static authorities return false because
         * they deliberately own no movement executable.
         */
        virtual bool deviceResidentMoEOverlayMaintenanceReady() const
        {
            return false;
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

        /**
         * @brief Archive live inference state at a scheduler-owned cursor.
         *
         * Production rollback is device-resident on GPU.  The caller therefore
         * supplies the exact logical cursor instead of asking the runner to
         * infer it from a potentially stale host shadow.
         */
        virtual PrefixStateSnapshot captureLivePrefixCheckpoint(
            const PrefixCheckpointCaptureRequest &request) const
        {
            (void)request;
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
