/**
 * @file DeviceGraphExecutor.h
 * @brief Compute graph execution engine
 * @author David Sanftenberg
 * @date December 2025
 *
 * DeviceGraphExecutor coordinates the execution of ComputeStages within a compute graph,
 * managing:
 * - Device-aware stage scheduling
 * - Asynchronous execution with dependency tracking
 * - Multi-device work distribution via WorkDistributor
 *
 * Note: This was previously named LayerExecutor but renamed to better reflect
 * its purpose - it executes graphs, not layers specifically.
 *
 * Architecture:
 *   Pipeline / ModelExecutor
 *      |
 *      v
 *   DeviceGraphExecutor  <-- This component
 *      |
 *      v
 *   ComputeStage[] (device-specific implementations)
 *      |
 *      v
 *   DeviceContext (execution environment)
 */

#pragma once

#include "IGraphExecutor.h"
#include "../../compute_stages/ComputeStages.h"
#include "../device/DeviceContext.h"
#include "../../mpi_orchestration/WorkDistributor.h"
#include "StageTimeline.h"
#include "../../../backends/DeviceId.h"
#include "../../../utils/DebugEnv.h" // For LLAMINAR_ASSERTIONS_ACTIVE
#include "../../../interfaces/ICollectiveContext.h"
#include "../../../backends/IGPUGraphCapture.h"
#include "../../../backends/IWorkerGPUContext.h"
#include "../coherence/StageCoherence.h" // For CoherenceBuffer (cached in GraphSegment)
#include "../../../memory/BufferArena.h" // Phase 2: contract-based coherence
#include <memory>
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <functional>

namespace llaminar2
{

    // Forward declarations
    class TensorBase;

    /**
     * @brief Represents a node in the compute graph
     */
    struct ComputeNode
    {
        std::string name;                      ///< Node identifier
        std::unique_ptr<IComputeStage> stage;  ///< The compute stage
        std::vector<std::string> dependencies; ///< Names of nodes this depends on
        DeviceId device;                       ///< Target device for execution
        bool completed;                        ///< Execution complete flag

        // =====================================================================
        // Coherence fast-path flags (mutable for use in const execution context)
        // Once weights are on-device and outputs are allocated, subsequent
        // iterations can skip the entire coherence check + vector extraction.
        // =====================================================================
        mutable bool weights_cohered = false; ///< Weights confirmed on device
        mutable bool is_final_output = false; ///< Outputs will be read by CPU (needs event sync)

        ComputeNode() : device(DeviceId::cpu()), completed(false) {}
        ComputeNode(std::string n, std::unique_ptr<IComputeStage> s, DeviceId dev = DeviceId::cpu())
            : name(std::move(n)), stage(std::move(s)), device(dev), completed(false) {}
    };

    /**
     * @brief Compute graph for execution
     *
     * A directed acyclic graph (DAG) of ComputeNodes with dependency tracking.
     * Enables parallel execution of independent nodes and proper ordering of
     * dependent operations.
     */
    class ComputeGraph
    {
    public:
        ComputeGraph() = default;
        ~ComputeGraph() = default;

        // Non-copyable
        ComputeGraph(const ComputeGraph &) = delete;
        ComputeGraph &operator=(const ComputeGraph &) = delete;

        // Movable
        ComputeGraph(ComputeGraph &&) = default;
        ComputeGraph &operator=(ComputeGraph &&) = default;

        /**
         * @brief Add a node to the graph
         * @param name Unique node identifier
         * @param stage The compute stage to execute
         * @param device Target device (DeviceId::cpu() for auto/CPU)
         * @return Reference to this graph for chaining
         */
        ComputeGraph &addNode(const std::string &name,
                              std::unique_ptr<IComputeStage> stage,
                              DeviceId device = DeviceId::cpu());

        /**
         * @brief Add a dependency between nodes
         * @param node_name The dependent node
         * @param depends_on The node that must complete first
         * @return Reference to this graph for chaining
         */
        ComputeGraph &addDependency(const std::string &node_name,
                                    const std::string &depends_on);

        /**
         * @brief Get execution order respecting dependencies
         * @return Reference to cached vector of node names in valid execution order
         *
         * The execution order is computed via topological sort on the first call
         * and cached until the graph is modified (addNode, addDependency, merge, clear).
         */
        const std::vector<std::string> &getExecutionOrder() const;

        /**
         * @brief Get nodes that can execute in parallel (no unmet dependencies)
         * @return Vector of names of ready nodes
         */
        std::vector<std::string> getReadyNodes() const;

        /**
         * @brief Get a node by name
         * @param name Node identifier
         * @return Pointer to node (nullptr if not found)
         */
        ComputeNode *getNode(const std::string &name);
        const ComputeNode *getNode(const std::string &name) const;

        /**
         * @brief Mark a node as completed
         * @param name Node identifier
         */
        void markCompleted(const std::string &name);

        /**
         * @brief Reset all completion flags
         */
        void reset();

        /**
         * @brief Check if all nodes are completed
         */
        bool allCompleted() const;

        /**
         * @brief Get number of nodes
         */
        size_t size() const { return nodes_.size(); }

        /**
         * @brief Get total estimated FLOPs for all stages
         */
        size_t totalEstimatedFlops() const;

        /**
         * @brief Clear the graph
         */
        void clear();

        /**
         * @brief Merge another graph into this one
         *
         * Moves all nodes from the source graph into this graph.
         * If connect_from is specified, adds dependencies from nodes in the source
         * graph that have no dependencies to the connect_from node in this graph.
         *
         * @param other The graph to merge (will be emptied)
         * @param connect_from Optional node name to connect source graph roots to
         * @return Reference to this graph for chaining
         */
        ComputeGraph &merge(ComputeGraph &&other, const std::string &connect_from = "");

        /**
         * @brief Get the names of all root nodes (nodes with no dependencies)
         * @return Vector of root node names
         */
        std::vector<std::string> getRootNodes() const;

        /**
         * @brief Get the names of all leaf nodes (nodes with no dependents)
         * @return Vector of leaf node names
         */
        std::vector<std::string> getLeafNodes() const;

        // =====================================================================
        // Fast Schedule — pre-computed flat array for zero-overhead decode loops
        // Eliminates string hash lookups, markCompleted calls, and virtual
        // type() dispatch from the per-token hot path.
        // =====================================================================

        struct FastScheduleEntry
        {
            ComputeNode *node;  ///< Direct pointer (no hash lookup needed)
            bool is_collective; ///< Pre-computed: ALLREDUCE / ALLGATHER / ALLGATHER_V
        };

        /**
         * @brief Build a pre-computed fast schedule from the execution order
         *
         * Must be called after the graph is finalized. The collective_nodes set
         * (if non-null) takes priority for collective classification; otherwise
         * falls back to stage->type().
         *
         * Also marks the last node as is_final_output for event-based dirty marking.
         */
        void buildFastSchedule(const std::unordered_set<std::string> *collective_nodes = nullptr);

        const std::vector<FastScheduleEntry> &fastSchedule() const { return fast_schedule_; }
        bool hasFastSchedule() const { return !fast_schedule_.empty(); }

    private:
        std::vector<std::unique_ptr<ComputeNode>> nodes_;
        std::unordered_map<std::string, size_t> node_index_;
        mutable std::vector<std::string> cached_order_; ///< Cached topological order
        mutable bool order_dirty_ = true;               ///< Invalidated on graph mutation
        std::vector<FastScheduleEntry> fast_schedule_;  ///< Pre-computed decode schedule
    };

    /**
     * @brief Orchestrates execution of compute stages within a graph
     *
     * DeviceGraphExecutor manages the execution of compute graphs by handling
     * topological ordering, device contexts, and execution modes.
     *
     * Usage:
     * @code
     * DeviceGraphExecutor executor(config);
     *
     * // Build a graph
     * ComputeGraph graph;
     * graph.addNode("norm", createRMSNormStage(...));
     * graph.addNode("proj", createGEMMStage(...));
     * graph.addDependency("proj", "norm");
     *
     * // Execute
     * executor.execute(graph, ctx);
     * @endcode
     */
    class DeviceGraphExecutor : public IGraphExecutor
    {
    public:
        /**
         * @brief Construct with configuration
         * @param config Executor configuration
         */
        explicit DeviceGraphExecutor(const GraphExecutorConfig &config = GraphExecutorConfig());
        ~DeviceGraphExecutor() override;

        // Non-copyable, movable
        DeviceGraphExecutor(const DeviceGraphExecutor &) = delete;
        DeviceGraphExecutor &operator=(const DeviceGraphExecutor &) = delete;
        DeviceGraphExecutor(DeviceGraphExecutor &&) = default;
        DeviceGraphExecutor &operator=(DeviceGraphExecutor &&) = default;

        // =========================================================================
        // Configuration (IGraphExecutor interface)
        // =========================================================================

        /**
         * @brief Get current configuration
         */
        const GraphExecutorConfig &config() const override { return config_; }

        /**
         * @brief Set execution mode
         */
        void setExecutionMode(ExecutionMode mode) override { config_.mode = mode; }

        /**
         * @brief Get current execution mode
         */
        ExecutionMode executionMode() const override { return config_.mode; }

        /**
         * @brief Enable/disable profiling
         */
        void setProfilingEnabled(bool enabled) override { config_.enable_profiling = enabled; }

        /**
         * @brief Enable/disable output validation
         */
        void setValidationEnabled(bool enabled) override { config_.enable_validation = enabled; }

        /**
         * @brief Set snapshot callback for debugging
         */
        void setSnapshotCallback(StageSnapshotCallback callback) override { config_.snapshot_callback = std::move(callback); }

        /**
         * @brief Set the current layer context for stage dumping
         */
        void setCurrentLayerIdx(int layer_idx) override { config_.current_layer_idx = layer_idx; }

        /**
         * @brief Set the current iteration context for stage dumping
         */
        void setCurrentIteration(int iteration) override { config_.current_iteration = iteration; }

        /**
         * @brief Set the MPI rank for stage dumping
         */
        void setMPIRank(int rank) override { config_.mpi_rank = rank; }

        // =========================================================================
        // Execution (IGraphExecutor interface)
        // =========================================================================

        /**
         * @brief Execute a compute graph
         * @param graph The compute graph to execute
         * @param ctx Device context for execution
         * @return true on success
         */
        bool execute(ComputeGraph &graph, IDeviceContext *ctx) override;

        /**
         * @brief Execute a compute graph with multi-device support
         * @param graph The compute graph to execute
         * @param contexts Map of DeviceId -> DeviceContext
         * @return true on success
         */
        bool executeMultiDevice(
            ComputeGraph &graph,
            const std::unordered_map<DeviceId, IDeviceContext *> &contexts) override;

        // =========================================================================
        // Statistics (IGraphExecutor interface)
        // =========================================================================

        /**
         * @brief Get execution statistics
         */
        const GraphExecutorStats &stats() const override { return stats_; }

        /**
         * @brief Reset statistics
         */
        void resetStats() override { stats_.reset(); }

        // =========================================================================
        // Collective Context
        // =========================================================================

        /**
         * @brief Set the collective context for GPU-native collectives
         *
         * When a collective context is set, ALLREDUCE and ALLGATHER stages
         * will be intercepted and executed via the CollectiveContext
         * infrastructure (RCCL, NCCL, PCIeBAR) instead of the stage's
         * internal MPI fallback.
         *
         * @param ctx Collective context (not owned, must outlive executor)
         */
        void setCollectiveContext(ICollectiveContext *ctx) { collective_ctx_ = ctx; }

        /**
         * @brief Get the current collective context
         * @return Pointer to collective context (nullptr if not set)
         */
        ICollectiveContext *collectiveContext() const { return collective_ctx_; }

        // =========================================================================
        // Buffer Management
        // =========================================================================

        /**
         * @brief Set the BufferArena for contract-based coherence
         *
         * When both the arena is set and a stage returns a non-empty bufferContract(),
         * the executor uses the arena for coherence instead of getDumpInfo().
         *
         * @param arena BufferArena (not owned, must outlive executor)
         */
        void setArena(BufferArena *arena) { arena_ = arena; }

        /**
         * @brief Get the current BufferArena
         * @return Pointer to arena (nullptr if not set)
         */
        BufferArena *arena() const { return arena_; }

        /**
         * @brief Execute a cached decode graph with minimal overhead
         *
         * Stripped-down execution path for cached decode graphs where:
         * - All tensors are already on the target GPU device
         * - Output GPU buffers are already allocated
         * - No stage dumping, validation, or profiling is needed
         * - Stage objects are reused (not rebuilt)
         *
         * Skips: getDumpInfo, extractBuffers, cohereInputs/Outputs,
         *        shouldDump, printStageOutputs, profiling, assertions
         *
         * The collective_nodes set enables O(1) lookup for TP stages instead
         * of calling stage->type() (virtual dispatch) on every node.
         *
         * @param graph The cached compute graph (stages already configured)
         * @param ctx Device context for execution
         * @param collective_nodes Pre-computed set of node names that are collective stages (optional, for TP>1)
         * @return true on success
         */
        /**
         * @brief Get the stage timeline for external collection/printing
         *
         * The timeline is populated during executeFastDecode() when
         * LLAMINAR_GPU_STAGE_TIMING=1. The caller is responsible for calling
         * collect() and printSummary() after the forward pass completes.
         */
        StageTimeline &stageTimeline() { return stage_timeline_; }
        const StageTimeline &stageTimeline() const { return stage_timeline_; }

        bool executeFastDecode(ComputeGraph &graph, IDeviceContext *ctx,
                               const std::unordered_set<std::string> *collective_nodes = nullptr);

        /**
         * @brief Execute a cached decode graph with GPU graph capture/replay
         *
         * On first call: captures all kernel launches into a GPU graph, instantiates
         * and launches it. On subsequent calls: re-captures, updates the executable
         * in-place (hipGraphExecUpdate/cudaGraphExecUpdate), and launches.
         *
         * Falls back to executeFastDecode() if:
         * - Graph capture/instantiation fails
         * - Update fails more than max_failures times consecutively
         * - Collective nodes are present (TP>1 with cross-device communication)
         *
         * @param graph The cached compute graph
         * @param ctx Device context for execution
         * @param capture GPU graph capture object (created once, reused across calls)
         * @param collective_nodes Pre-computed collective node names (for TP>1)
         * @param gpu_stream Opaque GPU stream pointer to assign to stages for dispatch
         * @return true on success
         */
        bool executeWithGraphCapture(ComputeGraph &graph, IDeviceContext *ctx,
                                     IGPUGraphCapture *capture,
                                     const std::unordered_set<std::string> *collective_nodes = nullptr,
                                     void *gpu_stream = nullptr);

        // =========================================================================
        // Segmented GPU Graph Capture/Replay
        // =========================================================================

        /**
         * @brief A segment of the execution graph — either graph-capturable or manual
         *
         * The execution order is partitioned into contiguous segments based on
         * isGraphCapturable(). Capturable segments get their own GPU graph;
         * non-capturable segments (attention, KV cache) are executed manually
         * between graph launches.
         */
        struct GraphSegment
        {
            std::vector<std::string> stage_names;          ///< Ordered stage names in this segment
            bool capturable = true;                        ///< Whether this segment can be graph-captured
            std::unique_ptr<IGPUGraphCapture> capture;     ///< GPU graph (only for capturable segments)
            std::vector<IComputeStage *> replay_callbacks; ///< Stages needing onGraphReplayed() (precomputed)
            uint64_t last_executed_step = 0;               ///< Last decode-step where this segment executed

            // ── Pre-cached coherence buffers (populated once during capture) ──
            // Eliminates per-decode-step getDumpInfo() + extractOutputBuffers()
            // + dynamic_cast overhead (~1352 vector allocs + 676 virtual calls
            // per step for Qwen2.5-7B with 338 capturable stages).
            std::vector<CoherenceBuffer> cached_all_output_buffers; ///< Flattened outputs of all stages
            bool replay_buffers_cached = false;                     ///< True after first post-launch caching
        };

        /**
         * @brief Persistent cache of graph segments for segmented capture/replay
         *
         * Built once on the first decode step, reused across subsequent steps.
         * Capturable segments are replayed via GPU graph launch; non-capturable
         * segments are executed manually each step.
         */
        struct GraphSegmentCache
        {
            std::vector<GraphSegment> segments;       ///< Ordered segments
            bool initialized = false;                 ///< Whether segments have been built
            bool needs_capture = false;               ///< True after warmup, before capture
            int consecutive_failures = 0;             ///< Segment-level failure counter
            uint64_t decode_step = 0;                 ///< Monotonic segmented-execution step counter
            void *capture_stream = nullptr;           ///< Locally-created blocking stream for capture/replay
            void *sync_event = nullptr;               ///< Cached event for GPU-side inter-stream sync
            IWorkerGPUContext *gpu_ctx_ref = nullptr; ///< GPU context for stream lifecycle (not owned)
            static constexpr int kMaxFailures = 4;    ///< Disable after N failures

            GraphSegmentCache() = default;
            ~GraphSegmentCache()
            {
                destroySyncEvent();
                destroyCaptureStream();
            }

            // Move-only (non-copyable due to stream/event ownership)
            GraphSegmentCache(GraphSegmentCache &&other) noexcept
                : segments(std::move(other.segments)),
                  initialized(other.initialized),
                  needs_capture(other.needs_capture),
                  consecutive_failures(other.consecutive_failures),
                  decode_step(other.decode_step),
                  capture_stream(other.capture_stream),
                  sync_event(other.sync_event),
                  gpu_ctx_ref(other.gpu_ctx_ref)
            {
                other.capture_stream = nullptr;
                other.sync_event = nullptr;
                other.gpu_ctx_ref = nullptr;
            }
            GraphSegmentCache &operator=(GraphSegmentCache &&other) noexcept
            {
                if (this != &other)
                {
                    destroySyncEvent();
                    destroyCaptureStream();
                    segments = std::move(other.segments);
                    initialized = other.initialized;
                    needs_capture = other.needs_capture;
                    consecutive_failures = other.consecutive_failures;
                    decode_step = other.decode_step;
                    capture_stream = other.capture_stream;
                    sync_event = other.sync_event;
                    gpu_ctx_ref = other.gpu_ctx_ref;
                    other.capture_stream = nullptr;
                    other.sync_event = nullptr;
                    other.gpu_ctx_ref = nullptr;
                }
                return *this;
            }
            GraphSegmentCache(const GraphSegmentCache &) = delete;
            GraphSegmentCache &operator=(const GraphSegmentCache &) = delete;

            void reset()
            {
                segments.clear();
                initialized = false;
                needs_capture = false;
                consecutive_failures = 0;
                decode_step = 0;
                destroySyncEvent();
                destroyCaptureStream();
            }

            /// Create a local blocking stream for graph capture via the GPU context.
            /// @param ctx GPU context that creates the stream (stored for cleanup)
            bool ensureCaptureStream(IWorkerGPUContext *ctx);

            /// Destroy the capture stream if it exists
            void destroyCaptureStream();

            /// Create or get the cached sync event for inter-stream dependencies
            bool ensureSyncEvent(IWorkerGPUContext *ctx);

            /// Destroy the cached sync event if it exists
            void destroySyncEvent();
        };

        /**
         * @brief Execute with segmented GPU graph capture/replay
         *
         * Partitions the execution order into segments based on stage
         * isGraphCapturable(). Capturable segments (GEMMs, norms, SwiGLU, etc.)
         * are captured into separate GPU graphs and replayed. Non-capturable
         * segments (attention, KV cache append) are executed manually between
         * graph launches.
         *
         * For a 28-layer Qwen2.5 model this produces ~57 graph segments + 56
         * manual segments per decode step, with each hipGraphLaunch costing
         * ~5-10μs (total ~300-600μs host overhead vs ~43ms without graphs).
         *
         * On first call: builds segment list, captures all capturable segments.
         * On subsequent calls: replays captured graphs, re-executes manual stages.
         *
         * @param graph The cached compute graph
         * @param ctx Device context for execution
         * @param segment_cache Persistent segment cache (built once, reused)
         * @param gpu_stream Opaque GPU stream pointer for kernel dispatch
         * @param gpu_ctx GPU context for creating new graph captures
         * @return true on success
         */
        bool executeWithSegmentedGraphCapture(ComputeGraph &graph, IDeviceContext *ctx,
                                              GraphSegmentCache &segment_cache,
                                              void *gpu_stream,
                                              IWorkerGPUContext *gpu_ctx,
                                              const std::unordered_set<std::string> *collective_nodes = nullptr,
                                              bool collectives_graph_capturable = false);

        /**
         * @brief Policy object for decode capture/replay execution mode selection
         */
        struct DecodeCapturePolicy
        {
            bool allow_fast_decode = true;
            bool allow_segmented_capture = false;
            bool collective_segmented_enabled = false;
            bool collectives_graph_capturable = false; ///< True when collective ops can be captured into GPU graph
            int max_segment_failures = 4;
        };

        /**
         * @brief Execute decode graph according to a single policy object
         *
         * Centralizes mode selection between segmented replay, fast decode, and
         * full executor fallback. When segmented replay fails, this method falls
         * back to fast decode automatically.
         */
        bool executeDecodeWithCapturePolicy(
            ComputeGraph &graph,
            IDeviceContext *ctx,
            GraphSegmentCache *segment_cache,
            void *gpu_stream,
            IWorkerGPUContext *gpu_ctx,
            const std::unordered_set<std::string> *collective_nodes,
            const DecodeCapturePolicy &policy,
            bool *used_segmented_capture = nullptr);

    private:
        GraphExecutorConfig config_;
        GraphExecutorStats stats_;
        ICollectiveContext *collective_ctx_ = nullptr; ///< Optional collective context (not owned)
        BufferArena *arena_ = nullptr;                 ///< Optional arena for contract coherence (not owned)
        StageTimeline stage_timeline_;                 ///< GPU event-based per-stage timeline profiler
        bool stage_timeline_info_populated_ = false;   ///< True after first setStageInfo pass (names never change)

        // Internal execution helpers
        bool executeSequential(ComputeGraph &graph, IDeviceContext *ctx);
        bool executeParallel(ComputeGraph &graph, IDeviceContext *ctx);
        bool executeNode(ComputeNode &node, IDeviceContext *ctx);

        // Collective stage intercept helpers (GPU-native collectives)
        bool executeCollectiveAllreduce(ComputeNode &node, IDeviceContext *ctx);
        bool executeCollectiveAllgather(ComputeNode &node, IDeviceContext *ctx);

        /**
         * @brief Execute strided AllGather using NCCL + CUDA deinterleave
         *
         * Optimized for column-parallel LM head. Returns false if NCCL not
         * available or device is not CUDA, allowing fallback to MPI path.
         *
         * @param node The compute node containing AllGatherStage
         * @param ctx Device context (unused, CollectiveContext handles device)
         * @return true if executed successfully, false to fall back to stage execution
         */
        bool executeCollectiveStridedAllgather(ComputeNode &node, IDeviceContext *ctx);

        // Buffer validation (Debug/Integration builds only)
#if LLAMINAR_ASSERTIONS_ACTIVE
        /**
         * @brief Verify stage inputs before execution (ENTRY validation)
         *
         * Called automatically BEFORE stage execution when assertions are active.
         * Checks all INPUT buffers for null, NaN, or Inf data.
         *
         * Can be disabled at runtime with LLAMINAR_VALIDATE_INPUTS=0.
         * Throws VerificationFailure on validation failure with full context.
         *
         * @param node The node whose inputs should be validated
         * @param layer_idx Current layer index for error reporting
         * @throws verification::VerificationFailure if validation fails
         */
        void verifyStageEntry(const ComputeNode &node, int layer_idx);

        /**
         * @brief Verify stage outputs after execution (EXIT validation)
         *
         * Called automatically AFTER stage execution when assertions are active.
         * Checks all OUTPUT buffers for null, NaN, Inf, or all-zero data.
         *
         * Can be disabled at runtime with LLAMINAR_VALIDATE_BUFFERS=0.
         * Throws VerificationFailure on validation failure with full context.
         *
         * @param node The node whose outputs should be validated
         * @param layer_idx Current layer index for error reporting
         * @throws verification::VerificationFailure if validation fails
         */
        void verifyStageExit(const ComputeNode &node, int layer_idx);

        /**
         * @brief Validate stage outputs for zero/NaN tensors (legacy)
         *
         * Called automatically after stage execution when assertions are active.
         * Checks all OUTPUT buffers for uninitialized (zero) or corrupted (NaN/Inf) data.
         *
         * Can be disabled at runtime with LLAMINAR_VALIDATE_BUFFERS=0.
         * NaN/Inf failure is enabled by default; zero-tensor warnings are not failures
         * unless LLAMINAR_FAIL_ON_ZERO=1 is set.
         *
         * @param node The node whose outputs should be validated
         * @return true if validation passes, false if errors detected
         *
         * @deprecated Use verifyStageExit() instead for exception-based validation
         */
        bool validateStageOutputs(const ComputeNode &node);
#endif

        // Workspace management
        std::vector<float> temp_buffer_;
        size_t temp_buffer_size_ = 0;

        float *getTemporaryBuffer(size_t elements);
    };

    // Backwards compatibility alias
    using LayerExecutor = DeviceGraphExecutor;

} // namespace llaminar2
