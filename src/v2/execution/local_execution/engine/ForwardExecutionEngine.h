/**
 * @file ForwardExecutionEngine.h
 * @brief Execution engine for forward graph dispatch and caching
 *
 * Extracted from DeviceGraphOrchestrator (Phase 3 of DGO refactor).
 *
 * This engine owns the forward graph cache and handles:
 * - Cache signature computation and lookup
 * - Cache HIT path: buffer updates, PP copy, dynamic params, GPU graph replay
 * - Cache MISS path: graph build via host callback, execution, cache population
 * - GPU stage timeline collection and printing
 *
 * The engine delegates model-specific operations (graph building, device context
 * management, logits sync) to an IForwardExecutionHost interface implemented by
 * the orchestrator.
 */

#pragma once

#include "ForwardGraphTypes.h"
#include "PrefillBucketUtils.h"
#include "../graph/DeviceGraphExecutor.h"
#include "../device/DeviceContext.h"
#include "../../factory/InferenceRunnerFactory.h" // For FactoryPPStageConfig
#include "../../../utils/ForwardPassProfiler.h"

#include <chrono>
#include <functional>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace llaminar2
{

    // Forward declarations
    class TensorBase;
    class GPUDeviceContextPool;

    /**
     * @brief Host interface for ForwardExecutionEngine callbacks
     *
     * The engine delegates model-specific and device-management operations
     * to the host (typically DeviceGraphOrchestrator). This interface defines
     * the minimal set of operations the engine cannot perform itself.
     *
     * Design rationale: virtual interface for clean abstraction boundary
     * and testability, even with a single production implementor.
     */
    class IForwardExecutionHost
    {
    public:
        virtual ~IForwardExecutionHost() = default;

        // ----- Graph Building (cache MISS path) -----

        /**
         * @brief Build a forward graph for the given input.
         *
         * The host dispatches internally to the appropriate builder
         * (standard / partial PP / unified PP).
         */
        virtual GraphBuildResult buildForwardGraph(const ForwardInput &input) = 0;

        // ----- Device Context -----

        /** Get or create a device context for the given device. */
        virtual IDeviceContext *getDeviceContext(DeviceId device) = 0;

        /** Get device contexts for all devices in a unified PP pipeline. */
        virtual std::unordered_map<DeviceId, IDeviceContext *> getPipelineDeviceContexts() = 0;

        // ----- Workspace -----

        /** Ensure GPU workspace is allocated for GEMM kernels in the graph. */
        virtual bool ensureDeviceWorkspaceAllocated(
            const ComputeGraph &graph,
            int workspace_seq_len = 0) = 0;

        /**
         * @brief Return the workspace generation for a device, if the host tracks it.
         *
         * A generation change means raw workspace addresses may have changed.
         * Cached graphs can still be reused, but captured GPU graph replay state
         * must be discarded and stages must be rebound before execution.
         */
        virtual uint64_t workspaceGeneration(DeviceId device) const
        {
            (void)device;
            return 0;
        }

        /**
         * @brief Called once after the first graph build completes and workspace
         *        is allocated, but before execution starts.
         *
         * Use this to release transient resources that are only needed during
         * graph construction (e.g., mmap pages via madvise(MADV_DONTNEED)).
         */
        virtual void onFirstGraphReady() {}

        // ----- Logits Synchronization -----

        /** Sync GPU stream and mark logits as host-readable. */
        virtual void syncLogitsAtBoundary(IDeviceContext *ctx) = 0;

        /** Access the logits tensor (for mapped-memory sync checks). */
        virtual TensorBase *logitsTensor() = 0;

        // ----- Decode Capture Policy -----

        /** Build the GPU graph capture/replay policy for decode steps. */
        virtual DeviceGraphExecutor::DecodeCapturePolicy buildDecodeCapturePolicy(
            bool has_collective_nodes,
            IDeviceContext *ctx,
            int segment_consecutive_failures) const = 0;

        // ----- PP Copy Info -----

        /**
         * @brief Information for copying external hidden state to working buffer.
         *
         * For PP non-embedding stages, the previous stage's output (external hidden)
         * must be copied to the local working buffer before graph execution. During
         * cache MISS this happens inline in graph build; during cache HIT the engine
         * must redo this copy using the stored PPCopyInfo.
         */
        struct PPCopyInfo
        {
            TensorBase *external_hidden = nullptr; ///< Source (stage N-1 output)
            TensorBase *working_buffer = nullptr;  ///< Destination (local residual/hidden)
            size_t copy_bytes = 0;
            DeviceId device;
            bool needs_copy = false;
        };

        /** Resolve PP copy info for the given input (after a cache-miss build). */
        virtual PPCopyInfo resolvePPCopyInfo(const ForwardInput &input) const = 0;

        /** Check if MoE dynamic rebalancing is active (blocks prefill graph capture). */
        virtual bool isMoeRebalancingActive() const { return false; }

        /** Whether the current forward graph must materialize logits for every input row. */
        virtual bool computeAllPositionLogitsEnabled() const { return false; }

        /** Compact verifier row count when all-position logits are row-indexed; 0 means full input rows. */
        virtual int allPositionLogitRows() const { return 0; }

        /**
         * @brief True while a vLLM-style MTP verifier row plan is installed.
         *
         * Batched verifier forwards have `batch_size > 1` and `seq_len > 1`,
         * so their shape alone looks like prefill. This explicit host signal
         * lets the engine route only those planned verifier forwards through
         * the decode graph cache and keeps ordinary batched prompt execution on
         * the prefill path.
         */
        virtual bool mtpSpecVerifierInputPlanActive() const { return false; }

        /**
         * @brief Prepare per-step metadata consumed by an all-position verifier graph.
         *
         * The forward engine calls this after workspace binding and after it has
         * chosen the stream that the graph will use for eager execution, warmup,
         * capture, or replay. Hosts can upload small graph-facing metadata here
         * without the engine knowing feature-specific details such as MTP row
         * selection. GPU hosts must treat `execution_stream` as mandatory when
         * `execution_device` is a GPU.
         */
        virtual bool prepareAllPositionVerifierGraphMetadata(
            const ForwardInput &input,
            void *execution_stream,
            DeviceId execution_device)
        {
            (void)input;
            (void)execution_stream;
            (void)execution_device;
            return true;
        }

        /**
         * @brief Order a forward graph after any pending live-state publication.
         *
         * MTP accepted-state publication can update KV, recurrent, and short-conv
         * state asynchronously on the verifier stream.  Before the next forward
         * reads that live state, the host gets the exact stream chosen for eager
         * execution, graph warmup, capture, or replay and may queue a GPU-side
         * wait.  This keeps publication atomic without forcing a CPU stream sync.
         */
        virtual bool prepareLiveStateForForwardGraphExecution(
            const ForwardInput &input,
            void *execution_stream,
            DeviceId execution_device)
        {
            (void)input;
            (void)execution_stream;
            (void)execution_device;
            return true;
        }

        /**
         * @brief Monotonic live-state epoch for decode graph replay safety.
         *
         * Speculative publication can jump KV/GDN/short-conv state by multiple
         * tokens. Ordinary one-token decode captures are only safe to replay
         * when they were captured after the current live-state epoch was
         * established, unless a future state-sync hook explicitly stamps them.
         */
        virtual uint64_t liveReplayStateEpoch() const { return 0; }

        /**
         * @brief Whether the next all-position verifier replay may defer its final sync.
         *
         * The host is responsible for allowing this only when the immediate
         * caller will consume all-position logits through a device operation on
         * the same replay stream. The default keeps the normal synchronized
         * forward boundary.
         */
        virtual bool shouldDeferAllPositionVerifierFinalSync() const { return false; }

        /**
         * @brief Receive the stream whose all-position verifier logits are pending.
         *
         * Called after a successful deferred replay. The host should enqueue the
         * next logits-consuming operation on this stream, or clear the pending
         * stream by passing nullptr when deferral is not active.
         */
        virtual void setPendingAllPositionVerifierStream(void *stream)
        {
            (void)stream;
        }

        /**
         * @brief Whether the next one-row main decode may defer its final sync.
         *
         * MTP condition-forward decode immediately consumes main logits with a
         * GPU sampler or GPU distribution-builder.  Deferring the replay sync
         * lets that consumer enqueue on the same stream instead of forcing the
         * CPU to wait between model forward and sampling.  Hosts should return
         * true only for one-shot decode paths where a device consumer is
         * guaranteed to follow.
         */
        virtual bool shouldDeferMainDecodeFinalSync() const { return false; }

        /**
         * @brief Receive the stream whose main-decode logits are pending.
         *
         * Called after a successful deferred main-decode replay.  The host must
         * enqueue the next logits-consuming device operation on this stream, or
         * clear the pending stream when nullptr is passed.
         */
        virtual void setPendingMainDecodeStream(void *stream)
        {
            (void)stream;
        }

        /** Domain placement epoch for MoE-sensitive prefill graph-cache keys. */
        virtual uint64_t moePlacementEpoch() const { return 0; }

        /** Domain id for MoE-sensitive prefill graph-cache observations. */
        virtual std::string prefillGraphDomainId() const { return "single"; }

        /** Domain-local participant id for MoE-sensitive prefill graph-cache observations. */
        virtual int prefillGraphParticipantId() const { return 0; }

        /** Stable topology fingerprint for MoE-sensitive prefill graph-cache observations. */
        virtual uint64_t prefillGraphTopologySignature() const { return 0; }

        /**
         * @brief Return safety state for optional chunk-boundary maintenance.
         *
         * Hosts with MoE rebalance domains can report histogram merge,
         * sparse-boundary, capture/replay, and participant-alignment state.
         * The default means no maintenance is requested for this chunk.
         */
        virtual PrefillChunkMaintenanceState prefillChunkMaintenanceState(
            const PrefillChunkPlan &chunk) const
        {
            PrefillChunkMaintenanceState state;
            state.chunk_index = chunk.chunk_index;
            state.histograms_merged = true;
            return state;
        }

        /**
         * @brief Run chunk-boundary maintenance after the gate allows it.
         *
         * This is where later MoE implementations merge telemetry-driven
         * rebalancing, prepared-weight transfer, runtime-table bank flips, and
         * graph-cache invalidation. The default is a no-op for non-MoE hosts.
         */
        virtual bool onPrefillChunkMaintenance(
            const PrefillChunkPlan &chunk,
            const PrefillChunkMaintenanceDecision &decision)
        {
            (void)chunk;
            (void)decision;
            return true;
        }
    };

    /**
     * @brief Forward graph execution engine with caching
     *
     * Manages the full lifecycle of forward graph execution:
     * 1. Compute cache signature from input + config
     * 2. On cache HIT: update buffers, copy PP hidden state, dispatch execution
     * 3. On cache MISS: build graph via host, execute, populate cache
     * 4. Collect GPU stage timeline telemetry
     *
     * Lifetime: owned by DeviceGraphOrchestrator, one per orchestrator instance.
     */
    class ForwardExecutionEngine
    {
    public:
        /**
         * @brief Static configuration for the engine (set once at construction).
         */
        struct Config
        {
            GraphCacheConfig cache_config;
            std::optional<FactoryPPStageConfig> pp_stage_config;
            bool has_unified_pp = false; ///< pipeline_config_ && pipeline_config_->hasPP()
        };

        ForwardExecutionEngine(Config config, DeviceGraphExecutor &executor);

        /**
         * @brief Runtime-facing host descriptor for one bucketed prefill chunk.
         *
         * The descriptor is intentionally pure: it selects a bucket, prepares
         * padded token/position buffers, and records whether padding would be
         * required. Launching remains gated by execute() and the prefill graph
         * cache preflight so unsafe padded graphs fail before execution.
         *
         * `ok` means the plan is allowed to execute under the current gate.
         * `chunk.ok` mirrors `ok` only for accepted plans; rejected padded plans
         * may still contain prepared buffers for diagnostics/tests.
         */
        struct PrefillChunkRuntimePlan
        {
            bool ok = false;                  ///< True when this chunk may execute under current gates.
            bool padding_required = false;    ///< True when real_count < bucket_seq_len.
            int chunk_index = 0;              ///< Stable chunk ordinal within a prepared schedule.
            bool rebalance_allowed_after = false; ///< True when a maintenance hook may run after this chunk.
            bool rebalance_required_after = false; ///< True when a maintenance hook must run after this chunk.
            PrefillBucketSelection selection; ///< Bucket selection result.
            PrefillChunkExecutionInput chunk; ///< Prepared host buffers and real/bucket metadata.
            std::string error;                ///< Failure reason when ok is false.

            /// @brief Convenience conversion for success checks.
            explicit operator bool() const { return ok; }
        };

        /**
         * @brief Prepared runtime plans for a multi-chunk prefill range.
         */
        struct PrefillChunkRuntimeSchedule
        {
            bool ok = false;
            PrefillChunkSchedule schedule;
            std::vector<PrefillChunkRuntimePlan> chunks;
            std::string error;

            explicit operator bool() const { return ok; }
        };

        /**
         * @brief Prepare one bucketed prefill chunk from a ForwardInput.
         *
         * @param input ForwardInput with stable buffers. `input.token_ids` must
         *        point to the first real token of the current chunk; `token_offset`
         *        supplies the absolute prompt offset for position IDs and is not
         *        added to the token pointer.
         * @param allow_padded_execution Leave false for callers that have not
         *        opted into fixed-bucket execution. Setting it true prepares
         *        padded buffers for runPrefillChunk(); graph safety is still
         *        enforced by PrefillGraphCache preflight before execution/capture.
         */
        static PrefillChunkRuntimePlan prepareSinglePrefillChunkRuntimePlan(
            const ForwardInput &input,
            const std::vector<int> &bucket_sizes,
            int pad_token_id,
            bool allow_padded_execution = false);

        /**
         * @brief Prepare all bucketed runtime chunks for an explicit schedule policy.
         *
         * `input.token_ids` must point to the first real token at
         * `input.token_offset`. The policy's real-token range must be contained
         * within `[input.token_offset, input.token_offset + input.real_seq_len)`;
         * `input.seq_len` is used when `input.real_seq_len` is not set.
         */
        static PrefillChunkRuntimeSchedule preparePrefillChunkRuntimeSchedule(
            const ForwardInput &input,
            const PrefillChunkSchedulerPolicy &policy,
            int pad_token_id,
            bool allow_padded_execution = false);

        /**
         * @brief Execute a forward pass (cache-aware).
         *
         * @param input  Prepared input (position_ids resolved, external_hidden applied)
         * @param output Receives logits/hidden pointers on success
         * @param host   Host interface for model-specific callbacks
         * @return true on success
         *
         * @pre input.position_ids != nullptr, or input.position_ids_device is
         *      set for a GPU graph whose position consumer supports resident
         *      device rows.
         * @pre external_hidden_state already set in input (if applicable)
         */
        bool execute(const ForwardInput &input,
                     ForwardOutput &output,
                     IForwardExecutionHost &host);

        struct LastExecutedForwardGraphView
        {
            ComputeGraph *graph = nullptr;
            ForwardGraphSignature signature;
            DeviceId device = DeviceId::invalid();
            void *stream = nullptr;
            bool cache_hit = false;
            bool is_decode = false;
            bool all_position_logits = false;

            explicit operator bool() const { return graph != nullptr; }
        };

        struct ReplayCacheObservation
        {
            ForwardGraphSignature signature;
            bool valid = false;
            bool segment_initialized = false;
            bool segment_needs_capture = false;
            bool phase3_active = false;
            bool gpu_stream_bindings_applied = false;
            bool has_capture_stream = false;
            uint64_t segment_decode_step = 0;
            uint64_t segmented_capture_live_state_epoch = 0;
            bool requires_live_state_epoch_recapture = false;
            bool all_position_verifier_recapture_pending = false;
        };

        /**
         * @brief Counts the replay-state action chosen for a live-state mutation.
         *
         * MTP publication is a precise boundary: live-state-versioned captures
         * must be recaptured before reuse, while single-row condition decode can
         * keep its executable and only rebind explicit streams. Returning this
         * summary makes that split testable and exportable.
         */
        struct ReplayStateResetSummary
        {
            size_t reset_replay_state = 0;
            size_t preserved_for_stream_rebind = 0;
            size_t ordinary_decode_reset = 0;
            size_t all_position_verifier_preserved = 0;
            size_t other_preserved = 0;
            bool all_position_verifier_recapture_requested = false;
        };

        /**
         * @brief Return the cached forward graph used by the most recent
         *        successful cached execution.
         *
         * MTP accepted-state publication needs the exact verifier graph that
         * just produced `draft_count + 1` target rows. This view is intentionally
         * narrow and returns empty for uncached or failed forwards so callers do
         * not accidentally publish from stale graph-cache entries.
         */
        std::optional<LastExecutedForwardGraphView> lastExecutedForwardGraph();

        /**
         * @brief Return the most recent all-position verifier graph.
         *
         * Accepted-state publication needs the graph whose GDN/short-conv stages
         * captured verifier row snapshots. A shifted-MTP sidecar commit may run
         * another graph before publication, so this handle is intentionally
         * separate from lastExecutedForwardGraph().
         */
        std::optional<LastExecutedForwardGraphView> lastAllPositionVerifierForwardGraph();

        /**
         * @brief Drop the retained verifier graph publication handle.
         *
         * Callers clear this after accepting, rejecting, or abandoning the
         * publication transaction so a later request cannot publish from stale
         * row snapshots.
         */
        void clearLastAllPositionVerifierForwardGraph();

        /**
         * @brief Snapshot replay-cache version/capture state for diagnostics/tests.
         *
         * The returned state is intentionally read-only. It lets unit and
         * integration guards prove that live-state mutations advance epochs and
         * force stale multi-row decode/verifier captures to recapture.
         */
        std::vector<ReplayCacheObservation> replayCacheObservations(
            uint64_t live_state_epoch) const;

        /**
         * @brief Execute one prepared bucketed prefill chunk.
         *
         * This Phase 6 boundary consumes the prepared chunk buffers from a
         * PrefillChunkRuntimePlan and delegates to execute() with fixed-bucket
         * shape metadata. Padded chunks are allowed only as prepared plans; the
         * forward graph path must pass PrefillGraphCache preflight before any
         * padded graph is executed, captured, or replayed.
         *
         * @param base_input ForwardInput carrying device, KV cache, PP, and
         *        batching context to preserve for the chunk execution.
         * @param plan Prepared runtime plan for the chunk. Must be exact.
         * @param output Receives logits/hidden pointers on success.
         * @param host Host interface for model-specific callbacks.
         * @return true when execute() succeeds; false if the plan is invalid,
         *         graph preflight rejects padded execution, or the delegated
         *         execution fails.
         */
        bool runPrefillChunk(
            const ForwardInput &base_input,
            const PrefillChunkRuntimePlan &plan,
            ForwardOutput &output,
            IForwardExecutionHost &host);

        /**
         * @brief Execute all chunks in a prepared prefill runtime schedule.
         *
         * Chunks run in schedule order. Each successful chunk passes through
         * the same chunk-boundary maintenance gate as runPrefillChunk(); the
         * first failed execution or maintenance hook stops the schedule.
         */
        bool runPrefillChunkSchedule(
            const ForwardInput &base_input,
            const PrefillChunkRuntimeSchedule &schedule,
            ForwardOutput &output,
            IForwardExecutionHost &host);

        // ----- Cache Management -----

        /** Invalidate all cached graphs and release resources. */
        void invalidateAll();

        /** Discard all cached forward graphs after a topology/workspace lifetime change. */
        void discardAllCachedGraphs();

        /**
         * @brief Drop captured GPU replay state while keeping cached ComputeGraphs.
         *
         * Live checkpoint restore rewinds KV/GDN/MTP state within an active request.
         * Cached graph objects and stable buffers are still usable, but replaying a
         * previously captured HIP/CUDA graph across that restore boundary can encode
         * stale runtime state. The next execution will warm up/capture again.
         */
        void resetCapturedReplayState();

        /**
         * @brief Drop only replay state that can be consumed by MTP correction replay.
         *
         * Rejected speculative steps publish an accepted verifier prefix, then
         * immediately replay the corrected token through the ordinary main
         * decode path. Any capture that encodes multi-row live-state progression
         * must be fresh. Dense single-token decode can preserve its executable
         * because every dynamic input is rebound before launch; MoE callers may
         * set @p preserve_single_token_decode_replay to false until routed
         * scratch/expert metadata preservation has its own equivalence proof.
         * Preserved caches still have their stage stream bindings dirtied so a
         * KernelFactory dynamic-state reset cannot leave backend kernels with
         * null streams before the next updateDynamicParams() call.
         */
        ReplayStateResetSummary resetCapturedReplayStateForCorrectionReplay(
            uint64_t live_state_epoch = 0,
            bool preserve_single_token_decode_replay = true);

        /**
         * @brief Drop captured all-position verifier replay after verifier-input mutation.
         *
         * Shifted MTP KV catch-up mutates an auxiliary cache read only by
         * multi-row verifier graphs.  Ordinary decode and sidecar captures do
         * not need to be discarded for that boundary, but ROCm verifier graph
         * replay has proven unsafe when only the live-state epoch advances.
         *
         * This method also records a one-shot recapture request for the next
         * all-position verifier graph.  That matters when the shifted-cache
         * mutation happens before the verifier cache exists, or while it is
         * between warmup/capture/replay phases: the next verifier execution
         * still has to settle on a freshly captured executable before the
         * request is consumed.
         */
        ReplayStateResetSummary resetAllPositionVerifierReplayState();

        /**
         * @brief Reset request/replay state without discarding cached forward graphs.
         *
         * Used at request/session boundaries after the orchestrator clears KV and
         * model recurrent state. Keeping the ComputeGraph avoids weight re-coherence;
         * resetting stage dynamic state and captured graph segments ensures the
         * next prompt starts from the newly cleared session state.
         */
        void resetSessionReplayState()
        {
            all_position_verifier_recapture_pending_ = false;
            for (auto &entry : cache_)
                entry.second.resetSessionState();
        }

        /** Check if cache is empty. */
        [[nodiscard]] bool cacheEmpty() const { return cache_.empty(); }

        /**
         * @brief Visit all stages of a given type across all cached graphs.
         * @param type   Stage type to filter for.
         * @param visitor Callback receiving (IComputeStage*) for each match.
         */
        void forEachCachedStage(ComputeStageType type,
                                const std::function<void(IComputeStage *)> &visitor) const;

        /**
         * @brief Immutable diagnostic snapshot for one cached prefill graph bucket.
         *
         * This is intentionally read-only: callers can observe the warmup,
         * capture, replay, and eviction counters without reaching into the
         * ForwardGraphCache internals or mutating graph lifetime state.
         */
        struct PrefillGraphCacheSnapshot
        {
            bool forward_cache_valid = false;                  ///< True when the matching forward graph entry is valid.
            bool prefill_cache_initialized = false;            ///< True after the prefill cache has been created for the entry.
            PrefillGraphPhase phase = PrefillGraphPhase::Cold; ///< Current bucket lifecycle phase.
            size_t cache_size = 0;                             ///< Number of bucket entries held by the prefill cache.
            size_t node_count = 0;                             ///< Captured graph node count for Ready entries.
            int replay_count = 0;                              ///< Successful graph launches for this bucket.
            uint64_t warmup_count = 0;                         ///< Lifetime warmups for this bucket.
            uint64_t capture_count = 0;                        ///< Lifetime successful captures for this bucket.
            uint64_t eviction_count = 0;                       ///< Total prefill bucket evictions observed by this engine.
            bool observation_valid = false;                    ///< True when runtime chunk/capture metadata has been observed.
            int chunk_index = 0;                               ///< Stable chunk ordinal from the latest prefill execution.
            int bucket_seq_len = 0;                            ///< Fixed graph bucket length from the latest prefill execution.
            int real_token_start = 0;                          ///< Inclusive real-token start offset from the latest prefill execution.
            int real_token_count = 0;                          ///< Real tokens represented by the latest prefill execution.
            int real_token_end = 0;                            ///< Exclusive real-token end offset from the latest prefill execution.
            std::string domain_id;                             ///< Prefix/MoE domain id associated with this graph observation.
            int participant_id = 0;                            ///< Domain-local participant id associated with this graph observation.
            uint64_t placement_epoch = 0;                      ///< MoE placement epoch associated with this graph observation.
            uint64_t topology_signature = 0;                   ///< Topology signature associated with this graph observation.
            std::string capture_phase;                         ///< cold, warmup, capture, replay, or rejected.
            std::string recapture_reason;                      ///< none or a structured reason for recapture/rejection.
        };

        /**
         * @brief Return diagnostic prefill graph cache state for a cached forward signature.
         *
         * Returns `std::nullopt` when the forward graph signature has not been
         * cached. A present snapshot may still report `prefill_cache_initialized=false`
         * because the first prefill request builds and caches the forward graph;
         * prefill graph warmup starts on the following cache hit.
         */
        std::optional<PrefillGraphCacheSnapshot> prefillGraphCacheSnapshot(
            const ForwardGraphSignature &signature,
            const PrefillGraphCacheKey &key) const;

        // ----- Mutable Execution Flags -----

        void setSuppressTimeline(bool v) { suppress_timeline_ = v; }
        void setAccumulatePrefill(bool v) { accumulate_prefill_ = v; }

        /** Access the forward pass profiler for flushing at benchmark end. */
        ForwardPassProfiler &forwardPassProfiler() { return forward_pass_profiler_; }

    private:
        struct LastExecutedForwardGraphState
        {
            bool valid = false;
            ForwardGraphSignature signature;
            bool cache_hit = false;
        };

        void recordLastExecutedForwardGraph(
            const ForwardGraphSignature &signature,
            bool cache_hit);
        std::optional<LastExecutedForwardGraphView> viewForLastExecutedForwardGraphState(
            const LastExecutedForwardGraphState &state);

        // ----- Cache HIT execution path -----
        bool executeCacheHit(
            const ForwardInput &input,
            ForwardOutput &output,
            ForwardGraphCache &cache,
            IForwardExecutionHost &host,
            bool is_decode,
            std::chrono::high_resolution_clock::time_point start);

        // ----- Cache MISS execution path -----
        bool executeCacheMiss(
            const ForwardInput &input_in,
            ForwardOutput &output,
            const ForwardGraphSignature &signature,
            ForwardGraphCache *build_cache,
            bool should_cache,
            IForwardExecutionHost &host,
            bool is_decode,
            bool has_unified_pp,
            std::chrono::high_resolution_clock::time_point start);

        // ----- Prefill graph capture/replay (Tier 1 Phase 5) -----
        bool executePrefillWithGraphCache(
            const ForwardInput &input,
            ForwardGraphCache &forward_cache,
            IDeviceContext *ctx,
            IForwardExecutionHost &host);

        /** @brief Mark a bucketed prefill forward-cache entry as recently used. */
        void touchBucketedPrefillForwardCache(
            const ForwardGraphSignature &signature,
            ForwardGraphCache &cache);

        /** @brief Enforce the engine-level cap for reusable bucketed prefill forward graphs. */
        void enforceBucketedPrefillForwardCapacity(
            const ForwardGraphSignature *active_signature = nullptr);

        /** @brief Count valid top-level bucketed prefill forward-cache entries. */
        size_t bucketedPrefillForwardCacheSize() const;

        // ----- GPU Stage Timeline collection -----
        void collectTimeline(
            IDeviceContext *ctx,
            bool is_decode,
            const ForwardInput &input,
            std::chrono::high_resolution_clock::time_point start,
            std::string stage_context = {});

        // ----- Configuration -----
        Config config_;
        DeviceGraphExecutor &executor_;

        // ----- Cache -----
        std::unordered_map<ForwardGraphSignature, ForwardGraphCache, ForwardGraphSignatureHash> cache_;
        uint64_t bucketed_prefill_forward_access_counter_ = 0;
        uint64_t bucketed_prefill_forward_eviction_count_ = 0;
        LastExecutedForwardGraphState last_executed_forward_graph_;
        LastExecutedForwardGraphState last_all_position_verifier_graph_;

        /*
         * Shifted MTP KV mutation is verifier-input mutation, not ordinary
         * live-prefix mutation.  A cache scan can drop existing verifier
         * captures, but it cannot protect a verifier graph that is created
         * after the mutation.  Keep a pending intent until an all-position
         * verifier has reached a fresh replay-ready capture.
         */
        bool all_position_verifier_recapture_pending_ = false;

        // ----- Mutable execution flags -----
        bool suppress_timeline_ = false;
        bool accumulate_prefill_ = false;
        bool first_graph_ready_fired_ = false;

        // ----- Forward pass wall-clock profiler -----
        ForwardPassProfiler forward_pass_profiler_;
    };

} // namespace llaminar2
