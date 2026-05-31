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

        /** Domain placement epoch for MoE-sensitive prefill graph-cache keys. */
        virtual uint64_t moePlacementEpoch() const { return 0; }

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
         * @pre input.position_ids != nullptr (caller must build if needed)
         * @pre external_hidden_state already set in input (if applicable)
         */
        bool execute(const ForwardInput &input,
                     ForwardOutput &output,
                     IForwardExecutionHost &host);

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

        /** Clear all cache entries (invalidate + remove). */
        void clearCache();

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
            std::chrono::high_resolution_clock::time_point start);

        // ----- Configuration -----
        Config config_;
        DeviceGraphExecutor &executor_;

        // ----- Cache -----
        std::unordered_map<ForwardGraphSignature, ForwardGraphCache, ForwardGraphSignatureHash> cache_;
        uint64_t bucketed_prefill_forward_access_counter_ = 0;
        uint64_t bucketed_prefill_forward_eviction_count_ = 0;

        // ----- Mutable execution flags -----
        bool suppress_timeline_ = false;
        bool accumulate_prefill_ = false;
        bool first_graph_ready_fired_ = false;

        // ----- Forward pass wall-clock profiler -----
        ForwardPassProfiler forward_pass_profiler_;
    };

} // namespace llaminar2
