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
#include "../graph/DeviceGraphExecutor.h"
#include "../device/DeviceContext.h"
#include "../../factory/InferenceRunnerFactory.h" // For FactoryPPStageConfig

#include <chrono>
#include <functional>
#include <optional>
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
        virtual bool ensureDeviceWorkspaceAllocated(const ComputeGraph &graph) = 0;

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

        // ----- Cache Management -----

        /** Invalidate all cached graphs and release resources. */
        void invalidateAll();

        /** Clear all cache entries (invalidate + remove). */
        void clearCache();

        /** Check if cache is empty. */
        [[nodiscard]] bool cacheEmpty() const { return cache_.empty(); }

        /**
         * @brief Visit all stages of a given type across all cached graphs.
         * @param type   Stage type to filter for.
         * @param visitor Callback receiving (IComputeStage*) for each match.
         */
        void forEachCachedStage(ComputeStageType type,
                                const std::function<void(IComputeStage*)>& visitor) const;

        // ----- Mutable Execution Flags -----

        void setSuppressTimeline(bool v) { suppress_timeline_ = v; }
        void setAccumulatePrefill(bool v) { accumulate_prefill_ = v; }

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

        // ----- Mutable execution flags -----
        bool suppress_timeline_ = false;
        bool accumulate_prefill_ = false;
        bool first_graph_ready_fired_ = false;
    };

} // namespace llaminar2
